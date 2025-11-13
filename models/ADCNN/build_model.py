from .classifier import LinearClassifierAlexNet
from .model import NetWork
from .LinearModel import alex_net_complete


def prepare_model(in_channel=1, feat_dim=1024,
                  n_hid_main=512, n_label=3, out_dim=12,
                  expansion=8, type_name='conv3x3x3', norm_type='Instance'):
    image_embeding_model = NetWork(in_channel=in_channel, feat_dim=feat_dim, expansion=expansion, type_name=type_name,
                                   norm_type=norm_type)

    # generate the classifier
    classifier = LinearClassifierAlexNet(in_dim=feat_dim, n_hid=n_hid_main, n_label=n_label)
    main_model = alex_net_complete(image_embeding_model, classifier)

    return main_model
