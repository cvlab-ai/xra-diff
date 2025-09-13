#!/bin/bash
source .cred

scp "$LOGIN@$IP:3dreconsnet/data/*.csv" data/
