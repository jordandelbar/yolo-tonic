#!/bin/bash
grpc_health_probe -addr=localhost:50051 || exit 1
exit 0
