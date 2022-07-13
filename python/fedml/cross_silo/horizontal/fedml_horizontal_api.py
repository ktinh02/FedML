from .fedml_aggregator import FedMLAggregator
from .fedml_trainer import FedMLTrainer
from .fedml_client_manager import FedMLClientManager
from .fedml_server_manager import FedMLServerManager
from .trainer.my_model_trainer_classification import (
    MyModelTrainer as MyModelTrainerCLS,
)
from .trainer.my_model_trainer_nwp import MyModelTrainer as MyModelTrainerNWP
from .trainer.my_model_trainer_tag_prediction import (
    MyModelTrainer as MyModelTrainerTAG,
)


def FedML_Horizontal(
    args,
    client_rank,
    client_num,
    comm,
    device,
    dataset,
    model,
    model_trainer=None,
    preprocessed_sampling_lists=None,
):
    [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ] = dataset
    if client_rank == 0:
        init_server(
            args,
            device,
            comm,
            client_rank,
            client_num,
            model,
            train_data_num,
            train_data_global,
            test_data_global,
            train_data_local_dict,
            test_data_local_dict,
            train_data_local_num_dict,
            model_trainer,
            preprocessed_sampling_lists,
        )
    else:
        init_client(
            args,
            device,
            comm,
            client_rank,
            client_num,
            model,
            train_data_num,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            model_trainer,
        )


def init_server(
    args,
    device,
    comm,
    client_rank,
    client_num,
    model,
    train_data_num,
    train_data_global,
    test_data_global,
    train_data_local_dict,
    test_data_local_dict,
    train_data_local_num_dict,
    model_trainer,
    preprocessed_sampling_lists=None,
):
    if model_trainer is None:
        model_trainer = get_model_trainer(args, model)
    model_trainer.set_id(0)

    # aggregator
    aggregator = FedMLAggregator(
        train_data_global,
        test_data_global,
        train_data_num,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        client_num,
        device,
        args,
        model_trainer,
    )

    # start the distributed training
    backend = args.backend
    if preprocessed_sampling_lists is None:
        server_manager = FedMLServerManager(
            args, aggregator, comm, client_rank, client_num, backend
        )
    else:
        server_manager = FedMLServerManager(
            args,
            aggregator,
            comm,
            client_rank,
            client_num,
            backend,
            is_preprocessed=True,
            preprocessed_client_lists=preprocessed_sampling_lists,
        )
    server_manager.run()


def init_client(
    args,
    device,
    comm,
    client_rank,
    client_num,
    model,
    train_data_num,
    train_data_local_num_dict,
    train_data_local_dict,
    test_data_local_dict,
    model_trainer=None,
):
    if model_trainer is None:
        model_trainer = get_model_trainer(args, model)
    model_trainer.set_id(client_rank)
    backend = args.backend
    trainer = FedMLTrainer(
        client_rank,
        train_data_local_dict,
        train_data_local_num_dict,
        test_data_local_dict,
        train_data_num,
        device,
        args,
        model_trainer,
    )
    client_manager = FedMLClientManager(
        args, trainer, comm, client_rank, client_num, backend
    )
    client_manager.run()


def get_model_trainer(args, model):
    if args.dataset == "stackoverflow_lr":
        model_trainer = MyModelTrainerTAG(model, args)
    elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
        model_trainer = MyModelTrainerNWP(model, args)
    else:  # default model trainer is for classification problem
        model_trainer = MyModelTrainerCLS(model, args)
    return model_trainer