import numpy as np

from movienet.tools import AudioExtractor

if __name__ == '__main__':
    extractor = AudioExtractor()

    features = []
    for i in range(3):
        wav_path = './tests/data/aud_wav/shot_{:04d}.wav'.format(i)
        feature = extractor.extract(wav_path)
        feature = feature.flatten()
        feature /= np.linalg.norm(feature)
        features.append(feature)

    features = np.stack(features)
    confuse_matrix = features.dot(features.T)
    print('confusion matrixs of stills:')
    print(confuse_matrix)
