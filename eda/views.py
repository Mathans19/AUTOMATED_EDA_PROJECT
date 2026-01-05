import os
import json
import pandas as pd
from io import StringIO
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_protect
from django.core.files.base import ContentFile
from .forms import UploadFileForm
from .models import Uploads, Reports, DatasetVersion, ChatMessage
# Removed ydata_profiling - using custom report generation
import plotly.express as px
import plotly.io as pio
from groq import Groq, RateLimitError
import numpy as np

# Monkey-patch numpy.asarray for compatibility
orig_asarray = np.asarray
def patched_asarray(a, dtype=None, order=None, **kwargs):
    kwargs.pop('copy', None)
    return orig_asarray(a, dtype=dtype, order=order, **kwargs)
np.asarray = patched_asarray

def calculate_health_score(df):
    """Calculate a data health score from 0-100"""
    if df.empty:
        return 0, []
        
    score = 100
    breakdown = []
    
    # 1. Missing Values penalty
    missing_pct = (df.isna().sum().sum() / (df.size)) * 100
    if missing_pct > 0:
        penalty = min(20, missing_pct * 0.5)
        score -= penalty
        breakdown.append(f"Missing Values: -{round(penalty, 1)} pts")
        
    # 2. Duplicate Rows penalty
    dup_pct = (df.duplicated().sum() / len(df)) * 100
    if dup_pct > 0:
        penalty = min(20, dup_pct * 1.0)
        score -= penalty
        breakdown.append(f"Duplicate Rows: -{round(penalty, 1)} pts")
        
    # 3. Constant Columns penalty
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        penalty = min(15, len(constant_cols) * 3)
        score -= penalty
        breakdown.append(f"Constant Columns: -{penalty} pts")
        
    # 4. Outliers penalty (numeric only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        outlier_total = 0
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = df[(df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))]
            if len(outliers) > 0:
                outlier_total += 1
        
        if outlier_total > 0:
            penalty = min(10, outlier_total * 2)
            score -= penalty
            breakdown.append(f"Outliers detected: -{penalty} pts")
            
    return max(0, round(score)), breakdown

def get_cleaning_suggestions(df):
    """Generate rule-based cleaning suggestions for each column"""
    suggestions = {}
    for col in df.columns:
        col_suggestions = []
        missing_pct = df[col].isna().mean() * 100
        dtype = df[col].dtype
        unique_pct = (df[col].nunique() / len(df)) * 100
        
        if missing_pct > 50:
            col_suggestions.append({"action": "drop_column", "reason": "Over 50% missing values", "severity": "high"})
        elif missing_pct > 0:
            if np.issubdtype(dtype, np.number):
                col_suggestions.append({"action": "fill_mean", "reason": "Numeric column has missing values", "severity": "medium"})
                col_suggestions.append({"action": "fill_median", "reason": "Numeric column has missing values", "severity": "medium"})
            else:
                col_suggestions.append({"action": "fill_mode", "reason": "Categorical column has missing values", "severity": "medium"})
        
        if df[col].nunique() == 1:
            col_suggestions.append({"action": "drop_column", "reason": "Constant value (no information)", "severity": "medium"})
            
        if not np.issubdtype(dtype, np.number) and df[col].nunique() < 10:
             col_suggestions.append({"action": "encode_onehot", "reason": "Few unique categories, suitable for encoding", "severity": "low"})
             
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
            
            # Create initial version
            DatasetVersion.objects.create(
                upload=upload,
                file=file,
                version_number=1,
                action_taken="Original Upload"
            )

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
        
        # Data Quality Health Score
        health_score, health_breakdown = calculate_health_score(df)
        
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
            'health_score': health_score,
            'health_breakdown': health_breakdown,
            'versions': upload.versions.all(),
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

        # Get health score and suggestions
        health_score, health_breakdown = calculate_health_score(df)
        suggestions = get_cleaning_suggestions(df)

        context = {
            "df": display_df.to_html(classes="table", index=False),
            "column_info": column_info,
            "file_id": file_id,
            "total_rows": len(df),
            "health_score": health_score,
            "health_breakdown": health_breakdown,
            "suggestions": suggestions,
            "versions": upload.versions.all(),
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

        # Save cleaned data as a NEW VERSION
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        
        action_desc = f"{action.replace('_', ' ').capitalize()}"
        if column:
            action_desc += f" on {column}"
        
        new_version_num = upload.current_version + 1
        new_version = DatasetVersion.objects.create(
            upload=upload,
            version_number=new_version_num,
            action_taken=action_desc
        )
        
        # Save file to the new version
        filename = f"{os.path.basename(upload.file.name).split('.')[0]}_v{new_version_num}.csv"
        new_version.file.save(filename, ContentFile(csv_buffer.getvalue().encode()), save=True)
        
        # Update current upload pointer
        upload.file = new_version.file
        upload.current_version = new_version_num
        upload.save()

        # Update column info after cleaning
        column_info = [
            (col, f"{df[col].isna().sum()} ({(df[col].isna().mean() * 100):.2f}%)", str(df[col].dtype))
            for col in df.columns
        ]
        
        # Recalculate health and suggestions
        health_score, health_breakdown = calculate_health_score(df)
        suggestions = get_cleaning_suggestions(df)

        return JsonResponse({
            "table_html": df.head(10000).to_html(classes="table", index=False),
            "column_info": column_info,
            "total_rows": len(df),
            "health_score": health_score,
            "health_breakdown": health_breakdown,
            "suggestions": suggestions,
            "success": True,
            "version_number": upload.current_version
        })

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON", "success": False}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e), "success": False}, status=500)


@require_POST
def undo_cleaning(request):
    try:
        data = json.loads(request.body)
        file_id = data.get("file_id")
        upload = get_object_or_404(Uploads, id=file_id)
        
        if upload.current_version <= 1:
            return JsonResponse({"error": "Already at the original version", "success": False}, status=400)
            
        # Get the current version and delete it
        current_v = DatasetVersion.objects.filter(upload=upload, version_number=upload.current_version).first()
        if current_v:
            current_v.delete()
            
        # Point to the previous version
        prev_v = DatasetVersion.objects.filter(upload=upload).order_by('-version_number').first()
        if prev_v:
            upload.file = prev_v.file
            upload.current_version = prev_v.version_number
            upload.save()
            
            # Load the data for preview
            df = pd.read_csv(upload.file.path)
            column_info = [
                (col, f"{df[col].isna().sum()} ({(df[col].isna().mean() * 100):.2f}%)", str(df[col].dtype))
                for col in df.columns
            ]
            
            health_score, health_breakdown = calculate_health_score(df)
            suggestions = get_cleaning_suggestions(df)

            return JsonResponse({
                "table_html": df.head(10000).to_html(classes="table", index=False),
                "column_info": column_info,
                "total_rows": len(df),
                "health_score": health_score,
                "health_breakdown": health_breakdown,
                "suggestions": suggestions,
                "success": True,
                "version_number": upload.current_version
            })
        else:
             return JsonResponse({"error": "No previous version found", "success": False}, status=400)

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
            
            # Get existing history
            history = ChatMessage.objects.filter(upload=upload).order_by('created_at')
            messages = [
                {"role": "system", "content": f"You are a senior data scientist. Analyzing dataset: {upload.name}. Shape: {df.shape}. Columns: {', '.join(df.columns)}"}
            ]
            
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})
            
            # Check if this is a request for a cleaning plan
            is_cleaning_plan = data.get("is_cleaning_plan", False)
            if is_cleaning_plan:
                user_question = "Generate a recommended cleaning plan for this dataset based on its structure and data types."
            
            # Add current question
            messages.append({"role": "user", "content": user_question})
            
            # Create a summary context for the AI if history is thin
            if len(history) < 2:
                buffer = StringIO()
                df.info(buf=buffer)
                info_str = buffer.getvalue()
                
                sum_context = f"""
                Data Info:
                {info_str}
                
                First 5 rows:
                {df.head().to_string()}
                
                Statistical Summary:
                {df.describe().to_string()}
                """
                messages[0]["content"] += f"\n\nContext:\n{sum_context}"

            chat_completion = client.chat.completions.create(
                messages=messages,
                model="llama-3.1-8b-instant",
            )
            
            ai_answer = chat_completion.choices[0].message.content
            
            # Save to history
            ChatMessage.objects.create(upload=upload, role="user", content=user_question)
            ChatMessage.objects.create(upload=upload, role="assistant", content=ai_answer)
            
            return JsonResponse({"answer": ai_answer, "success": True})

        except RateLimitError:
            return JsonResponse({"error": "Free tier rate limit reached. Please wait a minute and try again.", "success": False}, status=429)
        except Exception as e:
            return JsonResponse({"error": str(e), "success": False}, status=500)

    # Check if key is configured on server to infer UI state
    key_configured = bool(os.environ.get("GROQ_API_KEY"))
    chat_history = ChatMessage.objects.filter(upload=upload).order_by('created_at')
    return render(request, "eda/ai_insights.html", {
        "file_id": file_id,
        "key_configured": key_configured,
        "chat_history": chat_history
    })