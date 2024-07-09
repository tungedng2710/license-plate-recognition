import numpy as np

class BaseRecLabelDecode:
    def __init__(self, character_dict):
        self.character = self.add_special_char(character_dict + list('  '))
        self.dict = dict(enumerate(self.character))

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        for batch_idx in range(len(text_index)):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if remove_duplicate:
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list) if conf_list else np.nan))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class CTCLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self,character_dict):
        super().__init__(character_dict)
        self.char_mask = None

    def __call__(self, preds, label=None, *args, **kwargs):
        if self.char_mask is not None:
            preds[:, :, ~self.char_mask] = 0
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character

