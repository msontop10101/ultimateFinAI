# finance_app/serializers.py

from rest_framework import serializers

class QuestionSerializer(serializers.Serializer):
    message = serializers.CharField()
