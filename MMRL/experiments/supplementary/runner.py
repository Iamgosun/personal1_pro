"""Optional native trainer hooks: initialization snapshot, or E1 epoch snapshots.

No extra model forward or posterior draw; therefore no training RNG perturbation.
These extend RefactorRunner and leave all native validation/training unchanged.
"""


def register_trajectory():
    from dassl.engine import TRAINER_REGISTRY
    from trainers.refactor_runner import RefactorRunner
    names=(TRAINER_REGISTRY.registered_names() if hasattr(TRAINER_REGISTRY,'registered_names')
           else getattr(TRAINER_REGISTRY,'_obj_map',{}))
    if 'SupplementInitRunner' not in names:
        @TRAINER_REGISTRY.register()
        class SupplementInitRunner(RefactorRunner):
            def before_train(self):
                super().before_train()
                from pathlib import Path
                root=Path(self.cfg.OUTPUT_DIR)/'trajectory'
                # Lightweight checkpoint epoch=0; before any optimizer update.
                self.save_model(-1,str(root),is_best=False)
    else:
        SupplementInitRunner=TRAINER_REGISTRY.get('SupplementInitRunner')
    if 'SupplementTrajectoryRunner' not in names:
        @TRAINER_REGISTRY.register()
        class SupplementTrajectoryRunner(SupplementInitRunner):
            def after_epoch(self):
                super().after_epoch()
                from pathlib import Path
                root=Path(self.cfg.OUTPUT_DIR)/'trajectory'
                self.save_model(self.epoch,str(root),is_best=False)

    # E3 needs only two intermediate checkpoints (e.g. epochs 5 and 25),
    # rather than storing all 50. The snapshot is taken after the native epoch
    # lifecycle and does not perform an extra forward pass or sample from q.
    if 'SupplementSelectedRunner' not in names:
        @TRAINER_REGISTRY.register()
        class SupplementSelectedRunner(SupplementInitRunner):
            def after_epoch(self):
                super().after_epoch()
                import os
                from pathlib import Path
                raw=os.environ.get('DAVI_SUPPLEMENT_E3_SAVE_EPOCHS','')
                epochs={int(v) for v in raw.split(',') if v.strip()}
                if self.epoch+1 in epochs:
                    root=Path(self.cfg.OUTPUT_DIR)/'trajectory'
                    self.save_model(self.epoch,str(root),is_best=False)
