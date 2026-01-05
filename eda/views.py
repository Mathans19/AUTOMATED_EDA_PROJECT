import os
import json
import shutil
import pandas as pd
from io import StringIO
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_protect
from .forms import UploadFileForm
from .models import Uploads, Reports, ChatMessage, DatasetVersion
# Removed ydata_profiling - using custom report generation
import plotly.express as px
import plotly.io as pio
from groq import Groq, RateLimitError
import numpy as np

# Monkey-patch numpy.asarray for compatibility with some library dependencies
# that might pass 'copy' argument (NumPy 2.0+ style) to older NumPy versions.
orig_asarray = np.asarray
def patched_asarray(a, dtype=None, order=None, **kwargs):
    kwargs.pop('copy', None)
    return orig_asarray(a, dtype=dtype, order=order, **kwargs)
np.asarray = patched_asarray


def calculate_health_score(df):
    """Calculate a data health score from 0-100."""
    score = 100
    total_cells = df.size
    if total_cells == 0:
        return 0
    
    # 1. Missing values (Deduct up to 30 points)
    missing_pct = (df.isna().sum().sum() / total_cells) * 100
    score -= min(30, missing_pct)
    
    # 2. Duplicate rows (Deduct up to 20 points)
    dup_pct = (df.duplicated().sum() / len(df)) * 100 if len(df) > 0 else 0
    score -= min(20, dup_pct * 2)
    
    # 3. Constant columns (Deduct 5 points each, up to 15)
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    score -= min(15, len(constant_cols) * 5)
    
    # 4. Outliers - basic IQR check for numeric (Deduct up to 15 points)
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        total_outliers = 0
        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((numeric_df[col] < (Q1 - 1.5 * IQR)) | (numeric_df[col] > (Q3 + 1.5 * IQR))).sum()
            total_outliers += outliers
        outlier_pct = (total_outliers / total_cells) * 100
        score -= min(15, outlier_pct * 10)

    return max(0, min(100, round(score)))


def get_cleaning_suggestions(df):
    """Generate rule-based cleaning suggestions for each column."""
    suggestions = {}
    for col in df.columns:
        col_suggestions = []
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / len(df)) * 100
        nunique = df[col].nunique()
        dtype = str(df[col].dtype)
        
        # Rule: High missing values
        if missing_pct > 50:
            col_suggestions.append({
                "type": "warning",
                "message": f"High missingness ({missing_pct:.1f}%). Consider dropping.",
                "action": "drop_column"
            })
        elif missing_count > 0:
            if "float" in dtype or "int" in dtype:
                col_suggestions.append({
                    "type": "info",
                    "message": "Numeric missing values. Suggest Mean/Median imputation.",
                    "action": "fill_missing"
                })
            else:
                col_suggestions.append({
                    "type": "info",
                    "message": "Categorical missing values. Suggest Mode imputation.",
                    "action": "fill_missing"
                })
        
        # Rule: Constant values
        if nunique == 1:
            col_suggestions.append({
                "type": "warning",
                "message": "Constant value column. Safe to remove.",
                "action": "drop_column"
            })
        
        # Rule: Categorical suggestions
        if ("object" in dtype or "category" in dtype) and 1 < nunique < 15:
             col_suggestions.append({
                "type": "suggestion",
                "message": f"Low cardinality ({nunique} unique). Good candidate for Label/One-Hot encoding.",
                "action": "encode"
            })
             
        if col_suggestions:
            suggestions[col] = col_suggestions
            
    return suggestions


def sample_dashboard(request):
    """View to render the sample dashboard UI."""
    # Mock data for the sample
    context = {
        'file_id': 1, # Mock ID for links
    }
    return render(request, 'eda/sample_dashboard.html', context)


def upload_file(request):
    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = form.cleaned_data["file"]
            name = form.cleaned_data["name"] or file.name  # Auto-assign name if empty

            # Save to Uploads model
            upload = Uploads.objects.create(name=name, file=file)

            return redirect("eda_report", file_id=upload.id)
    else:
        form = UploadFileForm()

    return render(request, "eda/upload.html", {"form": form})


def generate_eda_report(request, file_id):
    upload = get_object_or_404(Uploads, id=file_id)
    
    try:
        # Read the latest cleaned CSV file
        df = pd.read_csv(upload.file.path)

        # Generate comprehensive statistics
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Dataset Overview
        dataset_info = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': len(numeric_cols),
            'categorical_columns': len(categorical_cols),
            'datetime_columns': len(datetime_cols),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
            'duplicate_rows': df.duplicated().sum(),
            'total_missing': df.isna().sum().sum(),
            'missing_percentage': (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
        }
        
        # Column Information
        column_info = []
        for col in df.columns:
            col_data = {
                'name': col,
                'dtype': str(df[col].dtype),
                'missing_count': int(df[col].isna().sum()),
                'missing_percentage': round((df[col].isna().sum() / len(df)) * 100, 2),
                'unique_count': df[col].nunique(),
                'unique_percentage': round((df[col].nunique() / len(df)) * 100, 2)
            }
            
            if col in numeric_cols:
                col_data.update({
                    'mean': round(df[col].mean(), 2) if not df[col].isna().all() else None,
                    'median': round(df[col].median(), 2) if not df[col].isna().all() else None,
                    'std': round(df[col].std(), 2) if not df[col].isna().all() else None,
                    'min': round(df[col].min(), 2) if not df[col].isna().all() else None,
                    'max': round(df[col].max(), 2) if not df[col].isna().all() else None,
                    'q25': round(df[col].quantile(0.25), 2) if not df[col].isna().all() else None,
                    'q75': round(df[col].quantile(0.75), 2) if not df[col].isna().all() else None,
                    'skewness': round(df[col].skew(), 2) if not df[col].isna().all() else None,
                })
            elif col in categorical_cols:
                top_values = df[col].value_counts().head(5).to_dict()
                col_data['top_values'] = top_values
                
            column_info.append(col_data)
        
        # Correlation Matrix (for numeric columns) - convert to JSON-serializable format
        correlation_matrix = None
        if len(numeric_cols) > 1:
            corr_df = df[numeric_cols].corr().round(3)
            correlation_matrix = {str(col): {str(c): float(corr_df.loc[col, c]) for c in corr_df.columns} 
                                 for col in corr_df.index}
        
        # Missing Values Analysis
        missing_analysis = df.isna().sum()
        missing_analysis = missing_analysis[missing_analysis > 0].sort_values(ascending=False)
        missing_dict = {col: {'count': int(count), 'percentage': round((count/len(df))*100, 2)} 
                       for col, count in missing_analysis.items()}
        
        # Data Quality Alerts
        alerts = []
        if dataset_info['duplicate_rows'] > 0:
            alerts.append({
                'type': 'warning',
                'message': f"Found {dataset_info['duplicate_rows']} duplicate rows",
                'action': 'remove_duplicates'
            })
        
        high_missing_cols = [col for col, info in missing_dict.items() if info['percentage'] > 50]
        if high_missing_cols:
            alerts.append({
                'type': 'error',
                'message': f"Columns with >50% missing values: {', '.join(high_missing_cols)}",
                'action': 'review_missing'
            })
        
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        if constant_cols:
            alerts.append({
                'type': 'info',
                'message': f"Constant columns (single value): {', '.join(constant_cols)}",
                'action': 'review_constant'
            })
        
        # Sample Data - convert to list of dicts with proper handling
        sample_df = df.head(10)
        sample_data = []
        for idx, row in sample_df.iterrows():
            row_dict = {}
            for col in df.columns:
                val = row[col]
                # Convert numpy types to Python native types
                if pd.isna(val):
                    row_dict[col] = None
                elif isinstance(val, (np.integer, np.floating)):
                    row_dict[col] = float(val) if isinstance(val, np.floating) else int(val)
                else:
                    row_dict[col] = str(val)
            sample_data.append(row_dict)
        
        # Convert to JSON for JavaScript
        correlation_json = json.dumps(correlation_matrix) if correlation_matrix else None
        numeric_cols_json = json.dumps(numeric_cols)
        
        context = {
            'file_id': file_id,
            'dataset_info': dataset_info,
            'column_info': column_info,
            'correlation_matrix': correlation_json,
            'missing_analysis': missing_dict,
            'alerts': alerts,
            'sample_data': sample_data,
            'numeric_cols': numeric_cols_json,
            'categorical_cols': categorical_cols,
        }
        
        return render(request, "eda/report.html", context)
    except Exception as e:
        return HttpResponse(f"Error generating report: {str(e)}", status=500)


def clean_data_view(request, file_id):
    upload = get_object_or_404(Uploads, id=file_id)

    try:
        # Read CSV file
        df = pd.read_csv(upload.file.path)

        # Extract column info
        column_info = [
            (col, f"{df[col].isna().sum()} ({(df[col].isna().mean() * 100):.2f}%)", str(df[col].dtype))
            for col in df.columns
        ]

        MAX_ROWS = 10000
        display_df = df.head(MAX_ROWS)
        
        # Calculate health score and suggestions
        health_score = calculate_health_score(df)
        suggestions = get_cleaning_suggestions(df)
        version_history = upload.versions.all()

        context = {
            "df": display_df.to_html(classes="table", index=False),
            "column_info": column_info,
            "file_id": file_id,
            "total_rows": len(df),
            "health_score": health_score,
            "suggestions": suggestions,
            "version_history": version_history
        }
        return render(request, "eda/clean_data.html", context)
    except Exception as e:
        return HttpResponse(f"Error loading data: {str(e)}", status=500)


@csrf_protect
@require_POST
def apply_cleaning_action(request):
    try:
        data = json.loads(request.body)
        column, action, file_id = data.get("column"), data.get("action"), data.get("file_id")

        if not all([action, file_id]): # Column might be optional for some actions like remove_duplicates
            return JsonResponse({"error": "Missing parameters", "success": False}, status=400)

        upload = get_object_or_404(Uploads, id=file_id)
        df = pd.read_csv(upload.file.path)

        # Apply cleaning actions
        if action == "drop_column":
            df.drop(columns=[column], inplace=True)
        elif action == "drop_na":
            df.dropna(subset=[column], inplace=True)
        elif action == "remove_duplicates":
            df.drop_duplicates(inplace=True)
        elif action == "rename_column":
            new_name = data.get("new_name")
            if new_name:
                df.rename(columns={column: new_name}, inplace=True)
            else:
                return JsonResponse({"error": "New name not provided", "success": False}, status=400)
        elif action == "convert_type":
            new_type = data.get("new_type")
            try:
                if new_type == "numeric":
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                elif new_type == "datetime":
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                elif new_type == "categorical":
                    df[column] = df[column].astype('category')
            except Exception as e:
                return JsonResponse({"error": f"Conversion failed: {str(e)}", "success": False}, status=400)
        elif action == "remove_outliers":
            if pd.api.types.is_numeric_dtype(df[column]):
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                df = df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]
            else:
                return JsonResponse({"error": "Outlier removal requires numeric column", "success": False}, status=400)
        elif action == "fill_mean" and pd.api.types.is_numeric_dtype(df[column]):
            df[column].fillna(df[column].mean(), inplace=True)
        elif action == "fill_median" and pd.api.types.is_numeric_dtype(df[column]):
            df[column].fillna(df[column].median(), inplace=True)
        elif action == "fill_mode":
            mode_value = df[column].mode()
            if not mode_value.empty:
                df[column].fillna(mode_value[0], inplace=True)
            else:
                return JsonResponse({"error": "Mode could not be determined", "success": False}, status=400)
        elif action == "fill_constant":
            # Fill missing values with a constant provided by the user
            value = data.get("value")
            df[column].fillna(value, inplace=True)
        elif action == "fill_ffill":
            df[column].fillna(method="ffill", inplace=True)
        elif action == "fill_bfill":
            df[column].fillna(method="bfill", inplace=True)
        elif action == "encode_onehot":
            # One-hot encode categorical column
            if not (pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_categorical_dtype(df[column])):
                return JsonResponse({"error": "One-hot encoding requires a categorical column", "success": False}, status=400)
            dummies = pd.get_dummies(df[column], prefix=column)
            df.drop(columns=[column], inplace=True)
            df = pd.concat([df, dummies], axis=1)
        elif action == "encode_label":
            # Simple label encoding
            if not (pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_categorical_dtype(df[column])):
                return JsonResponse({"error": "Label encoding requires a categorical column", "success": False}, status=400)
            categories = {cat: i for i, cat in enumerate(df[column].astype("category").cat.categories)}
            df[column] = df[column].map(categories)
        else:
            return JsonResponse({"error": "Invalid action", "success": False}, status=400)

        # Implementation of Versioning: Create a version before/after the change
        new_version_num = upload.current_version + 1
        
        # Professional naming convention
        action_slug = action.replace('_', '-')
        col_slug = column.replace(' ', '-').lower() if column else 'dataset'
        version_filename = f"v{new_version_num}_{action_slug}_{col_slug}.csv"
        
        version_relative_path = os.path.join('uploads', 'versions', version_filename)
        version_absolute_path = os.path.join(os.path.dirname(upload.file.path), 'versions', version_filename)
        
        # Ensure versions directory exists
        os.makedirs(os.path.dirname(version_absolute_path), exist_ok=True)
        
        # Save cleaned data back to the main file AND the version file
        df.to_csv(upload.file.path, index=False)
        df.to_csv(version_absolute_path, index=False)

        # Create version record
        DatasetVersion.objects.create(
            upload=upload,
            file=version_relative_path,
            version_number=new_version_num,
            action_taken=f"{action} on {column}" if column else action
        )
        
        # Update current version in upload
        upload.current_version = new_version_num
        upload.save()

        # Update column info after cleaning
        column_info = [
            (col, f"{df[col].isna().sum()} ({(df[col].isna().mean() * 100):.2f}%)", str(df[col].dtype))
            for col in df.columns
        ]

        return JsonResponse({
            "table_html": df.head(10000).to_html(classes="table", index=False),
            "column_info": column_info,
            "total_rows": len(df),
            "health_score": calculate_health_score(df),
            "suggestions": get_cleaning_suggestions(df),
            "current_version": upload.current_version,
            "success": True
        })

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON", "success": False}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e), "success": False}, status=500)


def download_cleaned_data(request, file_id):
    upload = get_object_or_404(Uploads, id=file_id)

    try:
        with open(upload.file.path, "rb") as f:
            response = HttpResponse(f.read(), content_type="text/csv")
            response["Content-Disposition"] = f'attachment; filename="cleaned_{upload.name}"'
            return response
    except Exception as e:
        return HttpResponse(f"Error downloading file: {str(e)}", status=500)


def visualize_data(request, file_id):
    upload = get_object_or_404(Uploads, id=file_id)
    
    try:
        df = pd.read_csv(upload.file.path)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        all_cols = df.columns.tolist()

        if request.headers.get('x-requested-with') == 'XMLHttpRequest' and request.method == "POST":
            data = json.loads(request.body)
            chart_type = data.get("chart_type")
            x_col = data.get("x_col")
            y_col = data.get("y_col")
            bins = data.get("bins")
            color_col = data.get("color_col")
            size_col = data.get("size_col")

            fig = None
            if chart_type == "histogram":
                fig = px.histogram(
                    df,
                    x=x_col,
                    nbins=bins or None,
                    color=color_col or None
                )
            elif chart_type == "box":
                fig = px.box(df, x=x_col, y=y_col) if y_col else px.box(df, y=x_col)
            elif chart_type == "scatter":
                if x_col and y_col:
                    fig = px.scatter(
                        df,
                        x=x_col,
                        y=y_col,
                        color=color_col or None,
                        size=size_col or None
                    )
                else:
                    return JsonResponse({"error": "Scatter plot requires both X and Y axes", "success": False}, status=400)
            elif chart_type == "heatmap":
                corr = df.select_dtypes(include=['number']).corr()
                fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
            elif chart_type == "missing_heatmap":
                # Visualize missingness as 0/1 heatmap
                missing = df.isna().astype(int)
                fig = px.imshow(
                    missing,
                    labels=dict(x="Columns", y="Rows", color="Missing"),
                    color_continuous_scale=[[0, "#10b981"], [1, "#ef4444"]],
                    title="Missing Value Heatmap"
                )
            elif chart_type == "bar":
                if y_col:
                    # Group by x_col and aggregate y_col
                    grouped = df.groupby(x_col)[y_col].mean().reset_index()
                    fig = px.bar(grouped, x=x_col, y=y_col, title=f"Average {y_col} by {x_col}")
                else:
                    value_counts = df[x_col].value_counts().head(20)
                    fig = px.bar(x=value_counts.index, y=value_counts.values, title=f"Frequency of {x_col}")
            elif chart_type == "line":
                if y_col:
                    grouped = df.groupby(x_col)[y_col].mean().reset_index().sort_values(by=x_col)
                    fig = px.line(grouped, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
                else:
                    return JsonResponse({"error": "Line plot requires both X and Y axes", "success": False}, status=400)
            elif chart_type == "pie":
                if y_col:
                    # Sum values of Y for each category in X
                    # Ensure Y is numeric
                    if y_col in numeric_cols:
                         grouped = df.groupby(x_col)[y_col].sum().reset_index()
                         fig = px.pie(grouped, values=y_col, names=x_col, title=f"Sum of {y_col} by {x_col}")
                    else:
                        return JsonResponse({"error": "Y axis for Pie chart must be numeric", "success": False}, status=400)
                else:
                    value_counts = df[x_col].value_counts().head(10)
                    fig = px.pie(values=value_counts.values, names=value_counts.index, title=f"Frequency of {x_col}")
            elif chart_type == "violin":
                if y_col:
                    fig = px.violin(df, x=x_col, y=y_col, title=f"Violin Plot: {y_col} by {x_col}")
                else:
                    fig = px.violin(df, y=x_col, title=f"Violin Plot: {x_col}")
            elif chart_type == "density":
                if x_col in numeric_cols:
                    fig = px.density_heatmap(df, x=x_col, y=y_col if y_col and y_col in numeric_cols else None, 
                                           title="Density Heatmap" + (f": {x_col} vs {y_col}" if y_col else f": {x_col}"))
                else:
                    return JsonResponse({"error": "Density plot requires numeric column", "success": False}, status=400)
            
            if fig:
                graph_html = pio.to_html(fig, full_html=False)
                return JsonResponse({"graph_html": graph_html, "success": True})
            else:
                return JsonResponse({"error": "Failed to generate chart", "success": False}, status=400)

        return render(request, "eda/visualize.html", {
            "file_id": file_id,
            "numeric_cols": numeric_cols,
            "all_cols": all_cols
        })
    except Exception as e:
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({"error": str(e), "success": False}, status=500)
        return HttpResponse(f"Error loading visualization: {str(e)}", status=500)


def ask_ai(request, file_id):
    upload = get_object_or_404(Uploads, id=file_id)
    
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user_question = data.get("question")
            api_key = data.get("api_key") or os.environ.get("GROQ_API_KEY")

            if not api_key:
                return JsonResponse({"error": "API Key is required (provided neither in request nor environment)", "success": False}, status=400)

            client = Groq(api_key=api_key)


            # Prepare context about the data
            df = pd.read_csv(upload.file.path)
            
            # Create a summary of the dataframe
            buffer = StringIO()
            df.info(buf=buffer)
            info_str = buffer.getvalue()
            
            # System context
            system_prompt = f"""You are a senior data scientist. Focus on accuracy and brevity.
            Dataset Info:
            File: {upload.name}
            Columns: {list(df.columns)}
            Shape: {df.shape}
            Stats: {df.describe().to_string()}
            """

            messages = [{"role": "system", "content": system_prompt}]
            
            # Add historical context (last 5 messages)
            recent_history = upload.chat_history.all().order_by('-created_at')[:5]
            for msg in reversed(recent_history):
                messages.append({"role": "user" if msg.role == 'user' else "assistant", "content": msg.content})

            # Add the current question
            messages.append({"role": "user", "content": f"Context summary: {info_str}\n\nQuestion: {user_question}"})
            
            chat_completion = client.chat.completions.create(
                messages=messages,
                model="llama-3.1-8b-instant",
            )
            
            answer = chat_completion.choices[0].message.content
            
            # Persist chat history
            ChatMessage.objects.create(upload=upload, role='user', content=user_question)
            ChatMessage.objects.create(upload=upload, role='assistant', content=answer)
            
            return JsonResponse({"answer": answer, "success": True})

        except RateLimitError:
            return JsonResponse({"error": "Free tier rate limit reached. Please wait a minute and try again.", "success": False}, status=429)
        except Exception as e:
            return JsonResponse({"error": str(e), "success": False}, status=500)

    # Fetch existing chat history
    chat_history = upload.chat_history.all()
    
    # Check if key is configured on server to infer UI state
    key_configured = bool(os.environ.get("GROQ_API_KEY"))
    return render(request, "eda/ai_insights.html", {
        "file_id": file_id, 
        "key_configured": key_configured,
        "chat_history": chat_history
    })


@csrf_protect
@require_POST
def undo_cleaning(request, file_id):
    try:
        upload = get_object_or_404(Uploads, id=file_id)
        
        # Get the previous version
        if upload.current_version <= 1:
            return JsonResponse({"error": "No more versions to undo to", "success": False}, status=400)
            
        previous_version_num = upload.current_version - 1
        previous_version = DatasetVersion.objects.filter(upload=upload, version_number=previous_version_num).first()
        
        if not previous_version:
             # If exact previous doesn't exist, try getting the latest one that's less than current
             previous_version = DatasetVersion.objects.filter(upload=upload, version_number__lt=upload.current_version).order_by('-version_number').first()
        
        if not previous_version:
            return JsonResponse({"error": "Previous version not found", "success": False}, status=404)

        # Revert main file to previous version
        shutil.copy2(previous_version.file.path, upload.file.path)
        
        # Update upload current version
        upload.current_version = previous_version.version_number
        upload.save()
        
        # Remove the latest version record (the one we just undid)
        DatasetVersion.objects.filter(upload=upload, version_number__gt=previous_version.version_number).delete()

        # Reload data to return new state
        df = pd.read_csv(upload.file.path)
        column_info = [
            (col, f"{df[col].isna().sum()} ({(df[col].isna().mean() * 100):.2f}%)", str(df[col].dtype))
            for col in df.columns
        ]

        return JsonResponse({
            "table_html": df.head(10000).to_html(classes="table", index=False),
            "column_info": column_info,
            "total_rows": len(df),
            "success": True,
            "message": f"Reverted to version {upload.current_version}"
        })

    except Exception as e:
        return JsonResponse({"error": str(e), "success": False}, status=500)