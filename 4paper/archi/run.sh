#!/bin/bash

dot -Tpdf ./iam.dot -o ./iam.pdf
dot -Tpdf ./n-mnist.dot -o ./n-mnist.pdf
dot -Tpdf ./quickdraw.dot -o quickdraw.pdf
dot -Tpdf ./physionet.dot -o physionet.pdf
