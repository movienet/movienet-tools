import os.path as osp

import requests


def download_file_from_google_drive(id, destination):

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, 'wb') as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = 'https://docs.google.com/uc?export=download'

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


if __name__ == '__main__':
    model_dict = {
        'cascade_rcnn_x101_64x4d_fpn': '1skKdtuqizKF35OEhuPUhf6VLoQOxlcOA',
        'retinanet_r50_fpn': '1aV1j1DUPyOZZtpzhyaeSDg7xx_Sr_xBu',
        'resnet50_csm': '1ft4gCZDY9aT86g3OS60uO8qqgAZiHiHH',
        'mtcnn': '1SqH8w4kWq-AH85WuwVyy3_q_f6r27BgL',
        'irv1_vggface2': '1VtheUyc1Zfk3jH8dI8CIdi-Iey_TNLyB',
        'resnet50_places365': '1qoESX0atxW7q4Merrho3jdJdKXjES14p',
        'ava_fast_rcnn_nl_r50_c4_1x_kinetics':
        '1_D3FupfmkPaSSUQ7JJpl9QCnWrYQqJ4W'
    }
    for filename, fileid in model_dict.items():
        print('download {} ...'.format(filename))
        save_path = './model/{}.pth'.format(filename)
        if osp.isfile(save_path):
            print('{} already exist.'.format(save_path))
        else:
            download_file_from_google_drive(fileid, save_path)
            print('{} download success.'.format(filename))
