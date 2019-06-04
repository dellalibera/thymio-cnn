from KidnappedThymio import KidnappedThymio
from Thymio_cnn import CNN
import argparse


# Author Alessio Della Libera and Andrea Bennati

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kidnapped Thymio')
    parser.add_argument('--action', help='Action',  choices=['random_walking', 'human', 'stop'], default='human')
    parser.add_argument('--name', help='Thymio Name', required=True)
    parser.add_argument('--save_image', help='Save Images while moving',  default=False)
    parser.add_argument('--path_save_image', help='Path to save images',  default="./")
    parser.add_argument('--predict', help='Predict Labels while moving',  default=False)
    parser.add_argument('--render', help='Render images while moving',  default=False)
    parser.add_argument('--record', help='Record a video while moving',  default=False)
    parser.add_argument('--path_model', help='Path of the saved model',  default=None)

    args = parser.parse_args()

    thymio_name = args.name
    action = args.action
    save_image = args.save_image
    predict = args.predict
    render = args.render
    record = args.record
    path_model = args.path_model
    path_save_image = args.path_save_image

    if not path_model:
        path_model = './model/model.pt'

    cnn = CNN()
    if predict:
        cnn.load_model(path=path_model)

    t = KidnappedThymio(
        thymio_name=thymio_name,
        room=path_save_image,
        cnn=cnn,
        save_image=save_image,
        predict=predict,
        render=render,
        record=record)

    t.turn_on_led(color=[0.0, 1.0, 0.0])

    if action == 'random_walking':
        t.random_walking()
    elif action == 'human':
        t.human_control()
    else:
        t.stop()

    print("Finish")

