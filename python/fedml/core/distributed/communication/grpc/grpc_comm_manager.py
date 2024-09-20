import csv
import grpc
import logging
import os
import pickle
import threading
import time

lock = threading.Lock()

from concurrent import futures
from fedml.core.mlops.mlops_profiler_event import MLOpsProfilerEvent
from typing import List

from ...communication.base_com_manager import BaseCommunicationManager
from ...communication.message import Message
from ...communication.observer import Observer
from ..constants import CommunicationConstants
from ..grpc import grpc_comm_manager_pb2_grpc, grpc_comm_manager_pb2

# Check Service or serve?
from ...communication.grpc.grpc_server import GRPCCOMMServicer


class GRPCCommManager(BaseCommunicationManager):
    MSG_ARG_KEY_SENDER_RANK = "sender_rank"
    MSG_ARG_KEY_SENDER_IP = "sender_ip"
    MSG_ARG_KEY_SENDER_PORT = "sender_port"

    def __init__(
            self,
            grpc_ipconfig_path,
            topic="fedml",
            client_rank=0,
            client_num=0,
            args=None
    ):

        self._topic = topic
        self._observers: List[Observer] = []
        self.grpc_ipconfig_path = grpc_ipconfig_path
        self.client_rank = client_rank
        self.client_id = self.client_rank
        self.client_num = client_num
        self.args = args

        if self.client_rank == 0:
            self.node_type = "server"
            logging.info("############# THIS IS FL SERVER ################")
        else:
            self.node_type = "client"
            logging.info("------------- THIS IS FL CLIENT ----------------")
        self.opts = [
            ("grpc.max_send_message_length", 1000 * 1024 * 1024),
            ("grpc.max_receive_message_length", 1000 * 1024 * 1024),
            ("grpc.enable_http_proxy", 0),
        ]
        self.grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=client_num),
            options=self.opts,
        )

        self.grpc_mappings = self._init_grpc_mappings()  # Load input mappings.
        if self.client_id not in self.grpc_mappings:
            # if no record exists for the current client id, then
            # default ip and rank to "0.0.0.0" and BASE + RANK.
            self.ip = "0.0.0.0"
            self.port = CommunicationConstants.GRPC_BASE_PORT + self.client_rank
            self.grpc_mappings[self.client_id] = (self.client_rank, self.ip, self.port)
        else:
            _, self.ip, self.port = self.grpc_mappings[self.client_id]

        self.grpc_servicer = GRPCCOMMServicer(
            self.ip,
            self.port,
            self.client_num,
            self.client_rank
        )
        grpc_comm_manager_pb2_grpc.add_gRPCCommManagerServicer_to_server(
            self.grpc_servicer, self.grpc_server
        )
        logging.info(os.getcwd())

        self.grpc_server.add_insecure_port("{}:{}".format(self.ip, self.port))

        self.grpc_server.start()
        # Wait for 100 milliseconds to make sure the grpc
        # server has started before proceeding.
        time.sleep(0.01)
        self.is_running = True
        logging.info("grpc server started. Listening on port " + str(self.port))

    def send_message(self, msg: Message):
        # Register the sender rank, ip and port attribute on the message.
        msg.add_params(GRPCCommManager.MSG_ARG_KEY_SENDER_RANK, self.client_rank)
        msg.add_params(GRPCCommManager.MSG_ARG_KEY_SENDER_IP, self.ip)
        msg.add_params(GRPCCommManager.MSG_ARG_KEY_SENDER_PORT, self.port)
        logging.info("sending msg = {}".format(msg.get_params_wout_model()))
        logging.info("pickle.dumps(msg) START")
        pickle_dump_start_time = time.time()
        msg_pkl = pickle.dumps(msg)
        MLOpsProfilerEvent.log_to_wandb({"PickleDumpsTime": time.time() - pickle_dump_start_time})
        logging.info("pickle.dumps(msg) END")

        receiver_id = msg.get_receiver_id()
        receiver_rank, receiver_ip, receiver_port = self.grpc_mappings[int(receiver_id)]
        channel_url = "{}:{}".format(receiver_ip, receiver_port)

        channel = grpc.insecure_channel(channel_url, options=self.opts)
        stub = grpc_comm_manager_pb2_grpc.gRPCCommManagerStub(channel)

        request = grpc_comm_manager_pb2.CommRequest()
        logging.info("sending message to {}".format(channel_url))

        request.client_id = self.client_id
        request.message = msg_pkl

        tick = time.time()
        stub.sendMessage(request)
        MLOpsProfilerEvent.log_to_wandb({"Comm/send_delay": time.time() - tick})
        logging.debug("sent successfully")
        channel.close()

    def add_observer(self, observer: Observer):
        self._observers.append(observer)

    def remove_observer(self, observer: Observer):
        self._observers.remove(observer)

    def handle_receive_message(self):
        self._notify_connection_ready()
        self.message_handling_subroutine()
        # Cannot run message_handling_subroutine in new thread
        # Related https://stackoverflow.com/a/70705165
        # thread = threading.Thread(target=self.message_handling_subroutine)
        # thread.start()

    def message_handling_subroutine(self):
        start_listening_time = time.time()
        MLOpsProfilerEvent.log_to_wandb({"ListenStart": start_listening_time})
        while self.is_running:
            if self.grpc_servicer.message_q.qsize() > 0:
                lock.acquire()
                busy_time_start_time = time.time()
                msg_pkl = self.grpc_servicer.message_q.get()
                logging.info("Unpickle START.")
                unpickle_start_time = time.time()
                msg = pickle.loads(msg_pkl)
                MLOpsProfilerEvent.log_to_wandb({"UnpickleTime": time.time() - unpickle_start_time})
                logging.info("Unpickle END.")
                msg_type = msg.get_type()

                sender_id = int(msg.get_sender_id())
                if sender_id not in self.grpc_mappings:
                    sender_rank = int(msg.get_params()[GRPCCommManager.MSG_ARG_KEY_SENDER_RANK])
                    sender_ip = str(msg.get_params()[GRPCCommManager.MSG_ARG_KEY_SENDER_IP])
                    sender_port = int(msg.get_params()[GRPCCommManager.MSG_ARG_KEY_SENDER_PORT])
                    self.grpc_mappings[sender_id] = (sender_rank, sender_ip, sender_port)

                for observer in self._observers:
                    _message_handler_start_time = time.time()
                    observer.receive_message(msg_type, msg)
                    MLOpsProfilerEvent.log_to_wandb({"MessageHandlerTime": time.time() - _message_handler_start_time})
                MLOpsProfilerEvent.log_to_wandb({"BusyTime": time.time() - busy_time_start_time})
                lock.release()
        time.sleep(0.0001)

        MLOpsProfilerEvent.log_to_wandb({"TotalTime": time.time() - start_listening_time})
        return

    def stop_receive_message(self):
        self.grpc_server.stop(None)
        self.is_running = False

    def notify(self, message: Message):
        msg_type = message.get_type()
        for observer in self._observers:
            observer.receive_message(msg_type, message)

    def _notify_connection_ready(self):
        msg_params = Message()
        msg_params.sender_id = self.client_rank
        msg_params.receiver_id = self.client_rank
        msg_type = CommunicationConstants.MSG_TYPE_CONNECTION_IS_READY
        for observer in self._observers:
            observer.receive_message(msg_type, msg_params)

    def _init_grpc_mappings(self):
        mappings = dict()
        csv_reader = csv.reader(open(self.grpc_ipconfig_path, "r"))
        next(csv_reader)  # skip header line
        for row in csv_reader:
            eid, rank, ip, port = row
            mappings[int(eid)] = (int(rank), str(ip), int(port))
        return mappings
