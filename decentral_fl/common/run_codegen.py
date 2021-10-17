"""Runs protoc with the gRPC plugin to generate messages and gRPC stubs."""
from grpc_tools import protoc

#grpc_tools.protoc -I./protos --python_out=. --grpc_python_out=. ./protos/monaifl.proto

protoc.main((
    '',
    '-I.',
    '--python_out=.',
    '--grpc_python_out=.',
    './monaifl.proto',
))
