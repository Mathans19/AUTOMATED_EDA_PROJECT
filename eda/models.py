from django.db import models
import os

class Uploads(models.Model):
    name = models.CharField(max_length=255, blank=True, null=True)
    file = models.FileField(upload_to='uploads/', blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    current_version = models.IntegerField(default=1)

    class Meta:
        db_table = 'uploads'
        verbose_name_plural = 'Uploads'

    def __str__(self):
        return self.name or f"Upload {self.id}"

class Reports(models.Model):
    upload = models.OneToOneField(Uploads, on_delete=models.CASCADE)
    html_content = models.TextField()
    generated_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'reports'
        verbose_name_plural = 'Reports'

class ChatMessage(models.Model):
    upload = models.ForeignKey(Uploads, on_delete=models.CASCADE, related_name='chat_history')
    role = models.CharField(max_length=10) # 'user' or 'ai'
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'chat_messages'
        ordering = ['created_at']

class DatasetVersion(models.Model):
    upload = models.ForeignKey(Uploads, on_delete=models.CASCADE, related_name='versions')
    file = models.FileField(upload_to='uploads/versions/')
    version_number = models.IntegerField()
    action_taken = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'dataset_versions'
        ordering = ['-version_number']
