def feedback(prediction):

    if 0<prediction<0.35:
        return "The model predicts that the audio clip is FAKE"
    elif 0.35<prediction<0.45:
        return "The model isn't sure but this audio clip is more likely to be FAKE"
    elif 0.45<prediction<0.55:
        return "The model is confused about this audio clip"
    elif 0.55<prediction<0.65:
        return "The model isn't sure but this audio clip is more likely to be REAL"
    elif 0.65<prediction<1:
        return "The model predicts that the audio clip is REAL"
    elif prediction==0:
        return "The model predicts that the audio clip is FAKE"
    elif prediction==0.35:
        return "The model predicts that the audio clip is FAKE"
    elif prediction==0.45:
        return "The model predicts that the audio clip is FAKE"
    elif prediction==0.55:
        return "The model is confused about this audio clip"
    elif prediction==0.65:
        return "The model predicts that the audio clip is REAL"
    elif prediction==1:
        return "The model predicts that the audio clip is REAL"