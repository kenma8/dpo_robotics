import argparse
import pickle

def main(args):
    # Load the “messed‐up” preferences
    with open(args.input_path, "rb") as f:
        old_prefs = pickle.load(f)

    model1_label = "Model 1"
    model2_label = "Model 2"

    corrected_prefs = []

    for idx, pref in enumerate(old_prefs):
        meta = pref["metadata"]
        left_lbl = meta["left_model"]
        right_lbl = meta["right_model"]
        orig_pref_lbl = meta["preferred_model"]

        # 1) Recover the original “choice” (1 = pressed LEFT, 2 = pressed RIGHT)
        #    In the old metadata, preferred_model == left_model  <=> they pressed 1
        #                         preferred_model == right_model <=> they pressed 2
        if orig_pref_lbl == left_lbl:
            orig_choice = 1
        elif orig_pref_lbl == right_lbl:
            orig_choice = 2
        else:
            raise ValueError(f"Entry {idx}: preferred_model ({orig_pref_lbl}) "
                             f"not equal to left_model ({left_lbl}) or right_model ({right_lbl}).")

        # 2) Reconstruct which trajectory was “left” vs “right” in the old dump:
        #    In old dump, if orig_choice == 1, then
        #        old chosen_obs = left_obs, old rejected_obs = right_obs
        #    If orig_choice == 2, then
        #        old chosen_obs = right_obs, old rejected_obs = left_obs
        if orig_choice == 1:
            left_obs      = pref["chosen_obs"]
            left_act      = pref["chosen_act"]
            right_obs     = pref["rejected_obs"]
            right_act     = pref["rejected_act"]
        else:  # orig_choice == 2
            left_obs      = pref["rejected_obs"]
            left_act      = pref["rejected_act"]
            right_obs     = pref["chosen_obs"]
            right_act     = pref["chosen_act"]

        # 3) Determine the human’s INTENDED preferred model:
        #    They pressed “1” whenever they wanted Model 1, and “2” whenever they wanted Model 2.
        if orig_choice == 1:
            intended_pref_model = model1_label
        else:
            intended_pref_model = model2_label

        # 4) Figure out where that intended model actually sat (left vs right).
        #    If left_lbl == intended_pref_model, then corrected_choice = 1; else = 2.
        if left_lbl == intended_pref_model:
            corrected_choice = 1
        elif right_lbl == intended_pref_model:
            corrected_choice = 2
        else:
            raise ValueError(
                f"Entry {idx}: intended_pref_model ({intended_pref_model}) "
                f"is neither left_label ({left_lbl}) nor right_label ({right_lbl})."
            )

        # 5) Build a brand‐new “chosen” vs “rejected” tuple based on corrected_choice:
        if corrected_choice == 1:
            new_chosen_obs    = left_obs
            new_chosen_act    = left_act
            new_rejected_obs  = right_obs
            new_rejected_act  = right_act
            new_pref_side     = "left"
        else:
            new_chosen_obs    = right_obs
            new_chosen_act    = right_act
            new_rejected_obs  = left_obs
            new_rejected_act  = left_act
            new_pref_side     = "right"

        # 6) Re‐assemble metadata with the corrected labels
        new_meta = {
            "pair_index":     meta["pair_index"],
            "env":            meta["env"],
            "left_model":     left_lbl,
            "right_model":    right_lbl,
            "preferred_side": new_pref_side,
            "preferred_model": intended_pref_model,
            "chosen_is_model1": (intended_pref_model == model1_label)
        }

        # 7) Push into a new preference entry
        corrected_entry = {
            "chosen_obs":    new_chosen_obs,
            "chosen_act":    new_chosen_act,
            "rejected_obs":  new_rejected_obs,
            "rejected_act":  new_rejected_act,
            "metadata":      new_meta
        }
        corrected_prefs.append(corrected_entry)

    # Finally, write out the “fixed” .pkl
    with open(args.output_path, "wb") as f_out:
        pickle.dump(corrected_prefs, f_out)

    print(f"✅  Loaded {len(old_prefs)} entries from '{args.input_path}'")
    print(f"✅  Wrote {len(corrected_prefs)} corrected entries to '{args.output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="“Fix” a human‐preference .pkl that was collected using „press 1 = Model 1, press 2 = Model 2“ "
                    "instead of „press 1 = left, press 2 = right“. "
                    "This script re‐computes chosen_obs/rejected_obs so that `chosen_obs` is always "
                    "the trajectory of the user’s intended model, and updates metadata accordingly."
    )
    parser.add_argument(
        "--input-path",     type=str, required=True,
        help="Path to the original (incorrectly‐labeled) pickle file."
    )
    parser.add_argument(
        "--output-path",    type=str, required=True,
        help="Where to write the corrected pickle file."
    )
    parser.add_argument(
        "--model1-label",   type=str, default="Model 1",
        help="The exact string label you passed as --model-1-label when collecting (usually 'Model 1')."
    )
    parser.add_argument(
        "--model2-label",   type=str, default="Model 2",
        help="The exact string label you passed as --model-2-label when collecting (usually 'Model 2')."
    )
    args = parser.parse_args()
    main(args)
