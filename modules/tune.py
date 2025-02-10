import optuna
from modules.model import AiModel


def objective(trial):
    learning_rate = trial.suggest_float("learning_rate",
                                                CONFIG['tunning']['learning_rate'][0],
                                                CONFIG['tunning']['learning_rate'][1],
                                                log=True
                                             )
    batch_size = trial.suggest_categorical("batch_size",
                                                CONFIG['tunning']['batch_size']
                                             )

    model = AiModel(batch_size=batch_size, learning_rate=learning_rate,
                    epochs=CONFIG['tunning']['epochs'])
    auc = model.fit()

    return auc

def run():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, CONFIG['tunning']['number_trails']) 

    print("Best Hyperparameters:", study.best_params)