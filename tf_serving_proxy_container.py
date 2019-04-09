from __future__ import print_function, absolute_import, division
import rpc
import os
import sys
import numpy as np
import logging
from datetime import datetime
import time
import threading
import bootstrap_server
import tensorflow as tf
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import random

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

## GRPC Helpers
class _ResultCounter(object):
    """Counter for the prediction results."""
    def __init__(self, batch_size):
        self.result_lst = ["" for _ in range(batch_size)]
        self.batch_size = batch_size
        self._condition = threading.Condition()
        self._result_count = 0
        self._ready_to_return = threading.Event()

    def save_result(self, i, result):
        with self._condition:
            self.result_lst[i] = result
            self._result_count += 1
            self._condition.notify()

            if self._result_count == self.batch_size:
                self._ready_to_return.set()
    
    def get_result(self):
        self._ready_to_return.wait()
        return self.result_lst

def _create_rpc_callback(i, result_counter):
    """Creates RPC callback function.
    Args:
        i: The position to insert result to
        result_counter: Counter for the prediction result.
    Returns:
        The callback function.
    """
    def _callback(result_future):
        """Callback function.
        Calculates the statistics for the prediction result.
        Args:
        result_future: Result future of the RPC.
        """
        exception = result_future.exception()
        if exception:
            result_counter.save_result(i, str(exception))
        else:
            response = str(result_future.result().outputs)
            result_counter.save_result(i, response)
    return _callback

class TFServingProxyContainer(rpc.ModelContainerBase):
    def __init__(self, grpc_endpoint, proc):
        self.channel = grpc.insecure_channel(grpc_endpoint)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
        self.proc = proc

    def predict_bytes(self, inputs):
        if self.proc.poll():
            raise Exception("The tf serving binary has terminated.")

        result_counter = _ResultCounter(len(inputs))
        for i, inp in enumerate(inputs):
            request = predict_pb2.PredictRequest()
            request.model_spec.name = "inception"
            request.model_spec.signature_name = "predict_images"
            request.inputs["images"].CopyFrom(
                tf.contrib.util.make_tensor_proto(inp.tobytes(), shape=[1])
            )
            result_future = self.stub.Predict.future(request, 5.0)  # 5 seconds
            result_future.add_done_callback(_create_rpc_callback(i, result_counter))
            sys.stdout.flush()
        return result_counter.get_result()

if __name__ == "__main__":
    try:
        model_name = os.environ["CLIPPER_MODEL_NAME"]
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_NAME environment variable must be set",
            file=sys.stdout,
        )
        sys.exit(1)

    try:
        model_version = os.environ["CLIPPER_MODEL_VERSION"]
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_VERSION environment variable must be set",
            file=sys.stdout,
        )
        sys.exit(1)

    ip = "127.0.0.1"
    if "CLIPPER_IP" in os.environ:
        ip = os.environ["CLIPPER_IP"]
    else:
        print("Connecting to Clipper on localhost")

    input_type = "bytes"
    if "CLIPPER_INPUT_TYPE" in os.environ:
        input_type = os.environ["CLIPPER_INPUT_TYPE"]

    bootstrap_kwargs = dict(
        max_batch_size=1, 
        batch_timeout_micros=1000, 
        max_enqueued_batches=200,
        port=8500,
        rest_api_port=8501,
        model_name="inception",
        model_base_path="/tf_models/inception"
    )
    for k,v in list(bootstrap_kwargs.items()):
        bootstrap_kwargs[k] = os.environ.get(k, v)

    # find a random port
    bootstrap_kwargs['port'] = random.randint(3000,40000)
    # disable http
    bootstrap_kwargs['rest_api_port'] = 0

    proc = bootstrap_server.start_tf_server(**bootstrap_kwargs)

    grpc_endpoint = "127.0.0.1:{port}".format(port=bootstrap_kwargs['port'])
    model = TFServingProxyContainer(grpc_endpoint, proc)
    rpc_service = rpc.RPCService()
    rpc_service.start(model, ip, model_name, model_version, input_type)