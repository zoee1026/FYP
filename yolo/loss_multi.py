import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from tensorflow.python.keras import backend as K
from config_multi import Parameters
from point_pillars import ciouraw
from box_class_utiliti import BBox, AnchorBBox
from box_utili import get_anchors_and_decode

class PointPillarNetworkLoss:

    def __init__(self, params: Parameters):
        self.alpha = float(params.alpha)
        self.gamma = float(params.gamma)
        self.focal_weight = float(params.focal_weight)
        self.loc_weight = float(params.loc_weight)
        self.size_weight = float(params.size_weight)
        self.angle_weight = float(params.angle_weight)
        self.heading_weight = float(params.heading_weight)
        self.class_weight = float(params.class_weight)
        self.anchor = params.anchor_dims
        self.anchors_mask = params.anchors_mask
        self.num_layer = len(params.anchors_mask)
        self.num_classes = params.nb_classes
        self.input_shape = [params.Xn, params.Yn]
        self.mapp = np.loadtxt("map.csv", delimiter=",")
        self.scale=params.scale

    def focal_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """ y_true value from occ in {-1, 0, 1}, i.e. {bad match, neg box, pos box} """

        self.mask = tf.equal(y_true, 1.)

        cross_entropy = K.binary_crossentropy(y_true, y_pred)

        p_t = y_true * y_pred + \
            (tf.subtract(1.0, y_true) * tf.subtract(1.0, y_pred))

        gamma_factor = tf.pow(1.0 - p_t, self.gamma)

        alpha_factor = y_true * self.alpha + \
            (1.0 - y_true) * (1.0 - self.alpha)

        focal_loss = gamma_factor * alpha_factor * cross_entropy

        neg_mask = tf.equal(y_true, 0)
        # changed percentile from 90 to 80
        thr = tfp.stats.percentile(tf.boolean_mask(focal_loss, neg_mask), 90.)
        hard_neg_mask = tf.greater(focal_loss, thr)
        # mask = tf.logical_or(tf.equal(y_true, 0), tf.equal(y_true, 1))
        mask = tf.logical_or(
            self.mask, tf.logical_and(neg_mask, hard_neg_mask))
        masked_loss = tf.boolean_mask(focal_loss, mask)

        return self.focal_weight*tf.reduce_mean(masked_loss)

    def _smooth_labels(self, y_true, label_smoothing):
        num_classes = tf.cast(K.shape(y_true)[-1], dtype=K.floatx())
        label_smoothing = K.constant(label_smoothing, dtype=K.floatx())
        return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

    def losses(self):
        return [self.loss_layer0,self.loss_layer1]
    
    def loss_layer0(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        return self.cal_loss(y_true,y_pred,0)
    
    def loss_layer1(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        return self.cal_loss(y_true,y_pred,1)

    def ciou_cal(self,y_true: tf.Tensor, y_pred: tf.Tensor):
        # conf, x, y, z, l, w, h, yaw, [classes]
        if len(y_pred)==0: return 0
        t=y_true.numpy()
        print(t.shape)
        p=y_pred.numpy()
        arr=np.concatenate((t, p),axis=-1)
        print('arr', arr.shape)
        iou =  np.apply_along_axis(lambda x: ciouraw(x[...,:7], x[...,7:]), axis=-1, arr=arr)
        print('iou',iou.shape)
        return K.variable(iou)
    
    def cal_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor, l=0):
        print(y_pred.shape, y_true.shape)
        # y_true = args[self.num_layers:]
        # y_pred = args[:self.num_layers]
        y_true=K.squeeze(y_true, axis=0)
        print(y_pred.shape, y_true.shape)
        loss = 0
        balance = [0.4, 1.0]
        object_mask = y_true[..., 0]
        print(object_mask.shape)

        num_pos = tf.maximum(K.sum(K.cast(object_mask, tf.float32)), 1)
        true_class_probs = y_true[..., 8:]
        true_class_probs = self._smooth_labels(true_class_probs, 0.01)
        grid, boxes, box_confidence, feats = get_anchors_and_decode(
            y_pred[0], self.anchor[self.anchors_mask[l]], self.num_classes, self.input_shape, self.mapp, self.scale[l], True)
        focal = self.focal_loss(y_true[..., 0], feats[..., 0])
        
        if np.count_nonzero(object_mask.numpy())!=0:
            ciou = self.ciou_cal(y_true[..., 1:7],
                            boxes[..., 1:7])
        else: ciou=K.constant(0)
        ciou_loss = object_mask * (1 - ciou)
        print(object_mask.shape,ciou.shape)

        tobj = tf.where(tf.equal(object_mask, 1), tf.maximum(
            ciou, tf.zeros_like(ciou)), tf.zeros_like(ciou))
        confidence_loss = K.binary_crossentropy(
            tobj, feats[..., 0], from_logits=True)
        
        object_mask=  K.expand_dims(object_mask, axis=-1)
        class_loss = object_mask * \
            K.binary_crossentropy(
                true_class_probs, feats[..., 8:], from_logits=True)
        class_loss = K.sum(class_loss) * 0.5 / num_pos / self.num_classes

        location_loss = K.sum(ciou_loss) * 0.05 / num_pos
        confidence_loss = K.mean(confidence_loss) * balance[l] * 1

        loss += focal + location_loss + confidence_loss + class_loss
        tf.Print(loss, [loss, location_loss,
                    confidence_loss, class_loss], message='loss: ')
        return loss
