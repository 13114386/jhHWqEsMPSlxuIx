from __future__ import unicode_literals, print_function, division

def import_model(args, options, vocabs, logger):
    try:
        modeling_choice = options.training.train_state["modeling_choice"]
    except Exception as ex:
        modeling_choice = None
    modeling_choice = args.modeling_choice if modeling_choice is None else modeling_choice
    assert modeling_choice == args.modeling_choice, \
            f"The specific {args.modeling_choice} is not compatible with the saved {modeling_choice}"
    if modeling_choice == "model_mrl":
        logger.info("Model is chosen from model.model_mrl")
        from model.model_mrl import Model
    elif modeling_choice == "model":
        logger.info("Model is chosen from model.model")
        from model.model import Model
    return Model(args, options, vocabs=vocabs, logger=logger)

def exclude_struct_features(struct_feature_level):
    if "struct" in struct_feature_level:
        excluded = {"encoder": ["coref"],
                    "decoder": ["coref"]}
    elif "coref" in struct_feature_level:
        excluded = {"encoder": [],
                    "decoder": ["coref"]}
    else:
        excluded = {"encoder": ["coref", "subword", "syntacdep"],
                    "decoder": ["coref", "subword", "syntacdep"]}
    return excluded
