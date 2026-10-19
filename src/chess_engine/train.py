from chess_engine.model import _train, _checkpoint_test
import os
import questionary


if __name__ == "__main__":
    opt1 = questionary.select(
        "What do you want to do?",
        choices=["Train model", "Test model", "Exit"], default="Train model"
    ).ask()

    if opt1 == "Exit":
        exit(-1)

    if opt1 == "Test model":
        opt1_1 = questionary.text(
            "Input path to model:"
        ).ask()
        assert os.path.exists(opt1_1), "File not existing."

        _checkpoint_test(checkpoint=opt1_1)

    if opt1 == "Train model":
        _train(
            epochs=4,
            n_train_shards=300,
            checkpoint_dir="models/test_model_2",
        )