# Run log

Distilled from the grid runner's output. One line per experiment, in the order they
executed. The raw logs are mostly progress bars and are not tracked; every number here
also lives in `results/<task>/<run_id>.json` with its full training history.

```
[grid] 16 runs
done agnews frozen_sk-logreg_cls_bert-base-uncased [2.0 min] {'accuracy': 0.88605, 'macro_f1': 0.88591, 'weighted_f1': 0.88591, 'n_examples': 7600}
done agnews frozen_sk-logreg_mean_bert-base-uncased [2.9 min] {'accuracy': 0.90184, 'macro_f1': 0.90177, 'weighted_f1': 0.90177, 'n_examples': 7600}
done agnews frozen_sk-linsvm_cls_bert-base-uncased [0.1 min] {'accuracy': 0.89211, 'macro_f1': 0.89204, 'weighted_f1': 0.89204, 'n_examples': 7600}
done agnews frozen_sk-rf_cls_bert-base-uncased [0.3 min] {'accuracy': 0.84868, 'macro_f1': 0.84791, 'weighted_f1': 0.84791, 'n_examples': 7600}
done agnews frozen_linear_bert-base-uncased [13.4 min] {'accuracy': 0.82092, 'macro_f1': 0.82042, 'weighted_f1': 0.82042, 'n_examples': 7600}
done agnews frozen_mlp_bert-base-uncased [15.7 min] {'accuracy': 0.84579, 'macro_f1': 0.84566, 'weighted_f1': 0.84566, 'n_examples': 7600}
done agnews partial_ft2_linear_bert-base-uncased [13.9 min] {'accuracy': 0.92355, 'macro_f1': 0.92347, 'weighted_f1': 0.92347, 'n_examples': 7600}
done agnews full_ft_linear_bert-base-uncased [17.4 min] {'accuracy': 0.93013, 'macro_f1': 0.93, 'weighted_f1': 0.93, 'n_examples': 7600}
done ner frozen_sk-logreg_cls_bert-base-uncased [1.2 min] {'entity_f1': 0.77842, 'entity_precision': 0.74723, 'entity_recall': 0.81232, 'token_accuracy': 0.96098, 'n_sentences': 3453, 'n_tokens': 46435}
done ner frozen_mlp_bert-base-uncased [4.4 min] {'entity_f1': 0.84175, 'entity_precision': 0.82398, 'entity_recall': 0.8603, 'token_accuracy': 0.97248, 'n_sentences': 3453, 'n_tokens': 46435}
done ner partial_ft4_linear_bert-base-uncased [4.8 min] {'entity_f1': 0.88097, 'entity_precision': 0.86736, 'entity_recall': 0.89501, 'token_accuracy': 0.97706, 'n_sentences': 3453, 'n_tokens': 46435}
done ner full_ft_linear_bert-base-uncased [10.9 min] {'entity_f1': 0.90413, 'entity_precision': 0.89588, 'entity_recall': 0.91254, 'token_accuracy': 0.98083, 'n_sentences': 3453, 'n_tokens': 46435}
done pos frozen_sk-logreg_cls_bert-base-uncased [1.6 min] {'token_accuracy': 0.93943, 'macro_f1': 0.88718, 'n_sentences': 2077, 'n_tokens': 25094}
done pos frozen_linear_bert-base-uncased [4.3 min] {'token_accuracy': 0.93269, 'macro_f1': 0.86767, 'n_sentences': 2077, 'n_tokens': 25094}
done pos partial_ft2_linear_bert-base-uncased [3.8 min] {'token_accuracy': 0.95656, 'macro_f1': 0.89064, 'n_sentences': 2077, 'n_tokens': 25094}
done pos full_ft_linear_bert-base-uncased [11.2 min] {'token_accuracy': 0.97513, 'macro_f1': 0.93703, 'n_sentences': 2077, 'n_tokens': 25094}
[grid] finished with 0 failure(s)
renamed ner/frozen_sk-logreg_cls_bert-base-uncased.json -> frozen_sk-logreg_firstsub_bert-base-uncased.json
renamed pos/frozen_sk-logreg_cls_bert-base-uncased.json -> frozen_sk-logreg_firstsub_bert-base-uncased.json
[migrate] 2 file(s)
[grid] 20 runs
skip agnews frozen_sk-logreg_cls_bert-base-uncased
skip agnews frozen_sk-logreg_mean_bert-base-uncased
skip agnews frozen_sk-linsvm_cls_bert-base-uncased
skip agnews frozen_sk-rf_cls_bert-base-uncased
skip agnews frozen_linear_bert-base-uncased
skip agnews frozen_mlp_bert-base-uncased
done agnews frozen_linear-pooler_bert-base-uncased [14.4 min] {'accuracy': 0.89605, 'macro_f1': 0.89586, 'weighted_f1': 0.89586, 'n_examples': 7600}
skip agnews partial_ft2_linear_bert-base-uncased
skip agnews full_ft_linear_bert-base-uncased
skip ner frozen_sk-logreg_firstsub_bert-base-uncased
done ner frozen_linear_bert-base-uncased [4.6 min] {'entity_f1': 0.80904, 'entity_precision': 0.78877, 'entity_recall': 0.83038, 'token_accuracy': 0.96576, 'n_sentences': 3453, 'n_tokens': 46435}
skip ner partial_ft4_linear_bert-base-uncased
skip ner full_ft_linear_bert-base-uncased
skip pos frozen_sk-logreg_firstsub_bert-base-uncased
skip pos frozen_linear_bert-base-uncased
skip pos partial_ft2_linear_bert-base-uncased
skip pos full_ft_linear_bert-base-uncased
done qa frozen_linear_bert-base-uncased [28.7 min] {'exact_match': 16.197, 'f1': 25.176, 'n_questions': 10570}
done qa partial_ft4_linear_bert-base-uncased [29.5 min] {'exact_match': 60.681, 'f1': 72.91, 'n_questions': 10570}
[grid] 3 runs
skip qa frozen_linear_bert-base-uncased
skip qa partial_ft4_linear_bert-base-uncased
```
