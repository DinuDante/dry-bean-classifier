"""Behavior checks for reproducibility, upload validation and inference-only UI."""
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest

from ml_core import (ROOT, FEATURES, TARGET, CLASSES, MODEL_FILES, METRICS,
                     evaluate, load_models, read_test_csv)


class ProjectChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload=(ROOT/"test_data.csv").read_bytes()
        cls.frame=read_test_csv(cls.payload)
        cls.models=load_models()
        cls.metrics=pd.read_csv(ROOT/"results/metrics.csv").set_index("ML Model Name")

    def test_saved_models_reproduce_all_reported_metrics(self):
        for name,model in self.models.items():
            actual,_=evaluate(model,self.frame)
            for metric in METRICS:
                self.assertAlmostEqual(actual[metric],self.metrics.loc[name,metric],places=12)
            self.assertEqual(list(model.feature_names_in_),FEATURES)
            self.assertEqual(list(model.classes_),CLASSES)

    def test_data_partition_is_disjoint_and_excludes_duplicate_measurements(self):
        raw=pd.read_csv(ROOT/"data/dry_bean.csv")
        clean=raw.drop_duplicates(subset=FEATURES).reset_index(drop=True)
        split=json.loads((ROOT/"results/split_indices.json").read_text())
        groups=[set(split[key]) for key in ("train","validation","test")]
        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        self.assertEqual(set.union(*groups),set(clean.index))
        pd.testing.assert_frame_equal(clean.loc[split["test"]].reset_index(drop=True)[FEATURES],
                                      self.frame[FEATURES],check_dtype=False)
        from train import candidates
        for name,model in self.models.items():
            if "standardscaler" in model.named_steps:
                fit_rows=clean.loc[sorted(groups[0]|groups[1]),FEATURES]
                np.testing.assert_allclose(model.named_steps["standardscaler"].mean_,fit_rows.mean().to_numpy())

    def test_upload_validation_and_column_reordering(self):
        reversed_frame=self.frame[list(reversed(self.frame.columns))]
        pd.testing.assert_frame_equal(read_test_csv(reversed_frame.to_csv(index=False).encode()),self.frame)
        invalid=[]
        invalid.append(self.frame.drop(columns=FEATURES[0]))
        invalid.append(self.frame.assign(unexpected=1))
        invalid.append(self.frame.iloc[:0])
        for value in (np.nan,np.inf,"not-a-number"):
            damaged=self.frame.head(2).astype(object)
            damaged.loc[damaged.index[0],FEATURES[0]]=value
            invalid.append(damaged)
        invalid.append(self.frame.head(2).assign(Class="UNKNOWN"))
        invalid.append(self.frame.head(2).assign(Class=None))
        invalid.append(pd.concat([self.frame]*3,ignore_index=True))
        for frame in invalid:
            with self.subTest(shape=frame.shape):
                with self.assertRaises(ValueError):
                    read_test_csv(frame.to_csv(index=False).encode())
        for payload in (b"",b"Area,Area,Class\n1,2,BOMBAY",b"\xff",b"a"*5242881):
            with self.assertRaises(ValueError):
                read_test_csv(payload)

    def test_incomplete_class_upload_keeps_defined_metrics_and_marks_auc_unavailable(self):
        one=self.frame[self.frame[TARGET]=="BOMBAY"].head(4)
        for model in self.models.values():
            scores,details=evaluate(model,one)
            self.assertIsNone(scores["AUC"])
            self.assertTrue(all(np.isfinite(scores[key]) for key in METRICS if key!="AUC"))
            self.assertEqual(sum(map(sum,details["confusion_matrix"])),4)

    def test_ui_all_models_and_upload_mode_never_fit(self):
        with patch.object(Pipeline,"fit",side_effect=AssertionError("The app must not train")):
            app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=90).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.metric),6)
            for name in MODEL_FILES:
                app.selectbox[0].set_value(name).run()
                self.assertFalse(app.exception)
                expected=self.metrics.loc[name]
                self.assertEqual([m.value for m in app.metric],[f"{expected[m]:.4f}" for m in METRICS])
            app.radio[0].set_value("Upload test CSV").run()
            self.assertFalse(app.exception)
            self.assertTrue(app.info)

    def test_bad_upload_is_shown_as_error_without_traceback(self):
        # AppTest versions differ in file-uploader support; mock only the input widget.
        from types import SimpleNamespace
        with patch("streamlit.file_uploader",return_value=SimpleNamespace(getvalue=lambda: b"bad\n1")):
            app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=90).run()
            app.radio[0].set_value("Upload test CSV").run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)

    def test_valid_upload_runs_all_models_without_fitting(self):
        from types import SimpleNamespace
        uploaded=self.frame.groupby(TARGET,sort=False).head(3)
        payload=uploaded.to_csv(index=False).encode()
        with patch("streamlit.file_uploader",return_value=SimpleNamespace(getvalue=lambda: payload)), patch.object(Pipeline,"fit",side_effect=AssertionError("Must not train")):
            app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=90).run()
            app.radio[0].set_value("Upload test CSV").run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            for name in MODEL_FILES:
                app.selectbox[0].set_value(name).run()
                scores,_=evaluate(self.models[name],read_test_csv(payload))
                self.assertEqual([m.value for m in app.metric],[f"{scores[m]:.4f}" for m in METRICS])


if __name__=="__main__":
    unittest.main(verbosity=2)
