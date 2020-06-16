# Digital Signal and Image Recognition project


# Team members
- Name: Gian Carlo, Surname: Milanese, Student ID: 848629, email: g.milanese1@campus.unimib.it
- Name: Khaled, Surname: Hechmi, Student ID: 793085, email: k.hechmi@campus.unimib.it

# Tasks
- Audio:
    - Digit recognition: from 0 to 9
    - Speaker recognition: among the two of us, our girlfriends and 4 speakers in the [free-spoken-digit dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)
- Images:
  - Face recognition: among the two of us and Gian Carlo's family members
- Retrieval:
  - Face similarity: find out which are the 10 VIPS that are more similar to us

# Project structure
The project is structured as follows:
- **[Audio](./Audio)**:
  - [0_record_audio.ipynb](./Audio/0_record_audio.ipynb) : this notebook can be used for quickly recording current user voice
  - [1_Data augmentation pipeline.ipynb](./Audio/1_data_augmentation_pipeline.ipynb): here we show the two augmentation strategies (random noise and pitch shift) we implemented and the empirical tests for finding the best values
  - [2_train_classifiers.ipynb](./Audio/2_train_classifiers.ipynb): this is the core notebook, where we load tracks, build predictive models and find out the best "model + data representation" for the two predictive tasks. The best models are stored in the directory [best_models](./Audio/best_models) for later reuse
  - [3_test_model_audio.ipynb](./Audio/3_test_model_audio.ipynb): here you can test the two best models we created
- **[Images](./Images)**:
  - [0_take_pictures.ipynb](./Images/0_take_pictures.ipynb): this notebook can be used for quickly taking pictures of the current user
  - [1_preprocess_pictures.ipynb](./Images/1_preprocess_pictures.ipynb): here we show the various preprocessing functions used for transforming input images so that predictive models will be more efficient
  - [2_train_model.ipynb](./Images/2_train_model.ipynb): this is the core notebook for the Image part, where we train various models in order to find out the best face recognition model. Best models are store in the [models directory](./Images/models)
  - [3_test_model_images.ipynb](./Images/3_test_model_images.ipynb): here you can test the best models and find out who you are most similar with.
  
We highly encourage you to **[have a look at our demo gifs](./Images/image_demos.md)** so that you can see how the model behaves while recognising different people!
- **[Retrieval](./Retrieval/)**: you will find the notebook [3_Image_retrieval.ipynb](3_Image_retrieval.ipynb) that can be used for finding which are the 10 VIP faces most similar to the current user

