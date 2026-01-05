from django import forms
from .models import Uploads

class UploadFileForm(forms.Form):
    name = forms.CharField(max_length=255, required=False)
    file = forms.FileField(required=True)

    def clean_name(self):
        """
        If no name is provided, use the original filename
        """
        name = self.cleaned_data.get('name')
        file = self.cleaned_data.get('file')
        
        if not name and file:
            return file.name
        return name

    def clean_file(self):
        """
        Validate uploaded file
        """
        file = self.cleaned_data.get('file')
        
        if not file:
            raise forms.ValidationError("Please select a file to upload.")
        
        # Check file extension
        if not file.name.lower().endswith('.csv'):
            raise forms.ValidationError("Only CSV files are allowed. Please upload a .csv file.")
        
        # Check file size (100MB limit)
        max_size = 100 * 1024 * 1024  # 100MB
        if file.size > max_size:
            raise forms.ValidationError(f"File size exceeds 100MB. Please upload a smaller file.")
        
        return file