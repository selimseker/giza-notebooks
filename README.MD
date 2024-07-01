- Model
    - In Giza Platform, a model represents a container for versions of your machine learning model
    - its a bit different from an actual ml model, its like a container for storing all cairo versions of your actual ML model
- transpilation
    - Transpilation is the process where you convert your machine learning model to a zkml model.
    - After transpilation, the zkstark model can be downloaded and executed locally
    - Models first need be converted to onnx format which is a standard way of representing machine learning models in many framework.
    
    ```bash
    # transpile the model to Cairo
    giza transpile uniswap_lp_nn_model.onnx --output-path uniswap_lp_nn
    # it will return model-id and version-id 
    ```
    
- version
    - a model in giza platform contains multiple versions
    - each version is a different sierra program
    - pick a model → create a model in giza platform → train model → transpile it to giza platform (add a new version to model in giza platform) → retrain your same model → transpile it to giza platform (add another version to model in giza platform)
- endpoint
    - Giza exposes the transpiled models through the endpoints.
    - There are 4 size types for the endoints depending on the model size (S, M, L, XL). These sizes utilize different amount of computational resources and request limits.
    
    ```bash
    # create an endpoint
    giza endpoints deploy --model-id 857 --version-id 3
    ```
    
- giza agent
    - Agents in giza are for taking some on-chain actions with the verified predictions.
    - Its basically an SDK for executing ZKML predictions and interacting with smart contracts.
    - It uses the [Ape](https://github.com/ApeWorX/ape) framework