Directory structure as follows. `inception` and `res50-models` are ommited to git. You can get them from s3://simon-mo-inferline/tfs-models
Run `make` to build all docker containers. 
Note that the `rpc.py` used has older protocol version that newer rpc so this is not compatible _yet_ with newest Clipper
```
.
├── bootstrap_server.py
├── clipper
│   ├── inception_container.py
│   ├── InceptionV3Dockerfile
│   └── Res50Dockerfile
├── inception
│   └── 1
│       ├── saved_model.pb
│       └── variables
│           ├── variables.data-00000-of-00001
│           └── variables.index
├── InceptionV3Dockerfile
├── __init__.py
├── Makefile
├── README.md
├── Res50Dockerfile
├── res50-models
│   └── 1
│       ├── saved_model.pb
│       └── variables
│           ├── variables.data-00000-of-00001
│           └── variables.index
├── rpc.py
├── TFServingBaseDockerfile
└── tf_serving_proxy_container.py

7 directories, 18 files
```
