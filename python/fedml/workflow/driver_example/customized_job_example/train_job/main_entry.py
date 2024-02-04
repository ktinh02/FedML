import os
import time
import traceback

import fedml

if __name__ == "__main__":
    print("Hi everyone, I am an launch job.")

    print(f"current config is {fedml.get_env_version()}")

    run_id = os.getenv('FEDML_CURRENT_RUN_ID', 0)
    edge_id = os.getenv('FEDML_CURRENT_EDGE_ID', 0)

    try:
        unique_id = "IMAGE_MODEL_KEY"
        my_api_key = ""  # Here you need to set your API key from nexus.fedml.ai
        response = fedml.api.download(unique_id,
                                      api_key=my_api_key, dest_path=f"{unique_id}")
        print(f"download response: code {response.code}, message {response.message}, data {response.data}")
    except Exception as e:
        print(f"download exception {traceback.format_exc()}")
        pass

    artifact = fedml.mlops.Artifact(name=f"general-file@{run_id}-{edge_id}", type=fedml.mlops.ARTIFACT_TYPE_NAME_GENERAL)
    artifact.add_file("./requirements.txt")
    artifact.add_dir("./config")
    fedml.mlops.log_artifact(artifact)

    fedml.mlops.log_model(f"model-file@{run_id}-{edge_id}", "requirements.txt")

    artifact = fedml.mlops.Artifact(name=f"log-file@{run_id}-{edge_id}", type=fedml.mlops.ARTIFACT_TYPE_NAME_LOG)
    artifact.add_file("./requirements.txt")
    artifact.add_dir("./config")
    fedml.mlops.log_artifact(artifact)

    artifact = fedml.mlops.Artifact(name=f"source-file@{run_id}-{edge_id}", type=fedml.mlops.ARTIFACT_TYPE_NAME_SOURCE)
    artifact.add_file("./requirements.txt")
    artifact.add_dir("./config")
    fedml.mlops.log_artifact(artifact)

    artifact = fedml.mlops.Artifact(name=f"dataset-file@{run_id}-{edge_id}", type=fedml.mlops.ARTIFACT_TYPE_NAME_DATASET)
    artifact.add_file("./requirements.txt")
    artifact.add_dir("./config")
    fedml.mlops.log_artifact(artifact)

    acc = 0.1
    loss = 2.0
    for iter_count in range(10):
        acc += 0.01
        loss -= 0.02
        fedml.mlops.log_metric({"acc": acc, "loss": loss})
        time.sleep(2)


    time.sleep(10)