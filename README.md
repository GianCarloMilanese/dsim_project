# Digital Signal and Image Recognition project

# Project structure
The project is structured as follows:
- **[Audio](./Audio)**: We chose to solve the following tasks: digit recognition (from 0 to 9) and speaker recognition (among the two of us, our girlfriends and 4 speakers in the [free-spoken-digit dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)). These are the main components:
  - [0_record_audio.ipynb](./Audio/0_record_audio.ipynb) : this notebook can be used for quickly recording digits
  - [1_Data augmentation pipeline.ipynb](./Audio/1_data_augmentation_pipeline.ipynb): here we show the two augmentation strategies (random noise and pitch shift) we implemented and the empirical tests for finding the best values
  - [2_train_classifiers.ipynb](./Audio/2_train_classifiers.ipynb): this is the core notebook, where we load tracks, build predictive models and find out the best "model + data representation" for the two predictive tasks. The best models are stored in the directory [best_models](./Audio/best_models) for later reuse
  - [3_test_model_audio.ipynb](./Audio/3_test_model_audio.ipynb): here you can test the two best models we created
- Images: the equivalent of the previous directory but for the Image domain.
- Retrieval: you will find the notebook for finding which are the 10 VIP faces most similar to the current user

# Team members
- Name: Gian Carlo, Surname: Milanese, Student ID: 848629, email: g.milanese1@campus.unimib.it
- Name: Khaled, Surname: Hechmi, Student ID: 793085, email: k.hechmi@campus.unimib.it
