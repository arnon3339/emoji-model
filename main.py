from modules.model import AiModel

if __name__ == "__main__":
    model = AiModel()
    model.fit()
    model.export()
    # print(model.get_dataset_for_tensor("train"))