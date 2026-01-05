from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_file, name='upload'),
    path('report/<int:file_id>/', views.generate_eda_report, name='eda_report'),
    path("clean/<int:file_id>/", views.clean_data_view, name="clean_data"),
    path("apply_cleaning/", views.apply_cleaning_action, name="apply_cleaning"),
    path("undo_cleaning/", views.undo_cleaning, name="undo_cleaning"),
    path("visualize/<int:file_id>/", views.visualize_data, name="visualize_data"),
    path("ask_ai/<int:file_id>/", views.ask_ai, name="ask_ai"),
    path("download/<int:file_id>/", views.download_cleaned_data, name="download_cleaned_data"),
    path("sample_dashboard/", views.sample_dashboard, name="sample_dashboard"),
]
