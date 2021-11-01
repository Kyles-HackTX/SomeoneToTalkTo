---
# S3T: Someone To Talk To

###### _Team Kyles: Suraj, Tejas, Lee, John, Utkarsh_

---

## Introduction
S3T is a powerful AI to detect the emotion from your voice! This full-stack application seeks to serve
business applications in customer-service for instance, where the customer's tone can tailor a better 
and more accustomed response from a company representative. Just speak into the microphone and the AI
will try to guess how you are feeling (and you can do this multiple times)!

## Key features
1. 45-50% accuracy on test sets, supporting both male and female voices
    - Trained on a limited dataset (RAVDESS) with only 2 sentences being spoken
    - Generalizes on ANY sentence (preferably in English)
    - Significantly better than a random guess, given 
    - Deployable for customer-service needs
    - Can be retrained for most tasks using transfer learning
    - Model completely made from scratch using a CNN-esque architecture
    - Unfortunately, a FFNN model architecture only yielded half the test accuracy, which is something we learned
2. Completely full-stack
    - Model in the cloud with a REST API linking the back-end
    - A friendly Electron-based front-end so not everything is a black-box
3. It's ML on the cloud!

## What we are competing for
1. HackTX overall prize, as we are a full-stack application
2. Likewize's speech-based emotion classification prize
3. Any others that the project may qualify for!

## Technologies used
- Electron
- PyTorch
- Kaggle
- Google Colab (& Cloud credits for training)
- Flask
- EC2

## How to run yourself
1. Run flask locally/publically using the built in web server.
2. Run the [electron app](https://github.com/Kyles-HackTX/electron-fe) locally as a frontend.
3. Talk into the microphone.

## Where we plan to go with this
We would like to continue working on this project after the contest as well, 
maybe commercializing after efforts to conduct more R&D. Perhaps crafting an AI
to formulate a response to the user's feelings may be another avenue!
