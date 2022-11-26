from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status

from .models import Project
from .serializers import *

@api_view(['GET', 'POST'])
def project_list(request):
    if request.method == 'GET':
        data = Project.objects.all()

        serializer = ProjectSerializer(data, context={'request': request}, many=True)

        return Response(serializer.data)

    elif request.method == 'POST':
        serializer = ProjectSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['PUT', 'DELETE'])
def project_detail(request, pk):
    try:
        project = Project.objects.get(pk=pk)
    except Project.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    if request.method == 'PUT':
        serializer = ProjectSerializer(project, data=request.data,context={'request': request})
        if serializer.is_valid():
            serializer.save()
            return Response(status=status.HTTP_204_NO_CONTENT)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    elif request.method == 'DELETE':
        project.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

"""@api_view(['GET'])
def getProjects(request):
    project = Project.objects.all()
    serializer = ProjectSerializer(project, many=True)
    return Response(serializer.data)

@api_view(['POST'])
def postProjects(request):
    data=request.data
    serializer = ProjectSerializer(data)
    if serializer.is_valid():
        serializer.save()
        return Response(status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['PUT'])
def putProjects(request, pk):
    data=request.data
    project = Project.objects.get(id=pk)
    serializer = ProjectSerializer(project, data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['DELETE'])
def delProjects(request, pk):
    project = Project.objects.get(id=pk)
    project.delete()
    return Response(status=status.HTTP_204_NO_CONTENT)"""