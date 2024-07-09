import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging
import onnxruntime as ort

from .model import Net


class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.onnx_inf = False
        if "onnx" in model_path:
            print("DeepSORT is running with ONNX runtime")
            self.onnx_inf = True
            self.sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.input_name = self.sess.get_inputs()[0].name
            self.label_name = self.sess.get_outputs()[0].name
        else:
            # print("DeepSORT is running without ONNX runtime")
            self.net = Net(reid=True)
            state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)[
                'net_dict']
            self.net.load_state_dict(state_dict)
            self.net.to(self.device)
            print("DeepSORT extractor is on:", next(self.net.parameters()).device)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(
            0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            if self.onnx_inf:
                im_batch = im_batch.cpu().numpy()
                # input_name = self.sess.get_inputs()[0].name
                # label_name = self.sess.get_outputs()[0].name
                features = self.sess.run([self.label_name], {self.input_name: im_batch})[0]
            else:
                im_batch = im_batch.to(self.device)
                features = self.net(im_batch).cpu().numpy()
        return features

# if __name__ == '__main__':
#     img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]
#     extr = Extractor("checkpoint/mars.t7")
#     feature = extr(img)
#     print(feature.shape)
