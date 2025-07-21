# train.py

from app.model import train_and_save_model

if __name__ == "__main__":
    train_and_save_model("data/Uber-Jan-Feb-FOIL.csv")
