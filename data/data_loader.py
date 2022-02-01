from data.paths import *

import os
import pathlib

import pandas as pd
import numpy as np

from PIL import Image


class MIMICDataloader():
    def __init__(self, XAMI_MIMIC_PATH) -> None:
        self.XAMI_MIMIC_PAHT = XAMI_MIMIC_PATH
        self.reflacx_metadata = pd.read_csv(TabularDataPaths.SpreadSheet.get_sreadsheet(
            self.XAMI_MIMIC_PAHT, TabularDataPaths.SpreadSheet.REFLACX.metadata))

    def get_tabular_data_path_from_field(self, patient_id, paths, reflacx_id=None):

        all_dfs = {}
        for path in paths:
            if "REFLACXStudy" in str(path):
                table_path = TabularDataPaths.PatientDataPaths.REFLACX.REFLACXStudy.get_reflacx_path(
                    self.XAMI_MIMIC_PAHT, patient_id, reflacx_id, path.value)
            else:
                table_path = TabularDataPaths.PatientDataPaths.get_patient_path(
                    self.XAMI_MIMIC_PAHT, patient_id, path)

            all_dfs[path] = pd.read_csv(table_path)

        return all_dfs

    def reflacx_get_ids_from_dicom_id(self, dicom_ids):
        id_map = {}

        found_instances = self.reflacx_metadata[self.reflacx_metadata['dicom_id'].isin(
            dicom_ids)]
        for _, instance in found_instances.iterrows():
            study_id = pathlib.Path(instance['image']).parts[-2][1:]
            id_map[instance['dicom_id']] = {
                "study_id": study_id,
                "patient_id": instance['subject_id'],
                "reflacx_id": instance['id']
            }

        return id_map

    def get_ids_from_dicom(self, dicom_ids):

        ids = []

        found_instances = self.reflacx_metadata[self.reflacx_metadata['dicom_id'].isin(
            dicom_ids)]

        for _, instance in found_instances.iterrows():
            study_id = pathlib.Path(instance['image']).parts[-2][1:]
            ids.append({
                "study_id": study_id,
                "patient_id": instance['subject_id'],
                "reflacx_id": instance['id'],
                "dicom_id": instance['dicom_id']
            })

        return ids

    def get_image_path(self, patient_id, study_id, dicom_id):
        return os.path.join(self.XAMI_MIMIC_PAHT, f"patient_{patient_id}", "CXR-JPG", f"s{study_id}", f"{dicom_id}.jpg")

    def get_reflacx_report_text_path(self, patient_id, reflacx_id):
        return os.path.join(self.XAMI_MIMIC_PAHT, f"patient_{patient_id}", "REFLACX", reflacx_id, "transcription.txt")

    def get_reflacx_eye_tracking_path(self,  patient_id, reflacx_id):
        return os.path.join(self.XAMI_MIMIC_PAHT, f"patient_{patient_id}", "REFLACX", reflacx_id, "fixations.csv")

    def get_relfacx_eye_gaze_path(self, reflacx_id):
        return os.path.join(self.XAMI_MIMIC_PAHT, "spreadsheets", "REFLACX", "gaze_data", reflacx_id, "gaze.csv")

    def get_cxr_report_text_path(self, patient_id, study_id):
        return os.path.join(self.XAMI_MIMIC_PAHT, f"patient_{patient_id}", "CXR-DICOM", f"s{study_id}")

    def get_data(self, dicom_ids, tabular_data_paths, load_image=True, load_report_text="reflacx"):

        all_data = {}

        ids_map = self.reflacx_get_ids_from_dicom_id(dicom_ids)

        for dicom_id in dicom_ids:

            all_data[dicom_id] = {}
            dicom_id_map = ids_map[dicom_id]
            all_data[dicom_id]['tabular'] = self.get_tabular_data_path_from_field(
                dicom_id_map['patient_id'], tabular_data_paths, reflacx_id=dicom_id_map['reflacx_id'])

            if load_image == True:
                image_path = self.get_image_path(
                    dicom_id_map['patient_id'], dicom_id_map['study_id'], dicom_id)
                all_data[dicom_id]['image'] = np.asarray(
                    Image.open(image_path))

            if str(load_report_text).lower() == "reflacx":
                report_text_path = self.get_reflacx_report_text_path(
                    dicom_id_map['patient_id'], dicom_id_map['reflacx_id'])
                all_data[dicom_id]['reflacx_text'] = pathlib.Path(
                    report_text_path).read_text()

            elif str(load_report_text).lower() == 'cxr':
                report_text_path = self.get_cxr_report_text_path(
                    dicom_id_map['patient_id'], dicom_id_map['study_id'])
                all_data[dicom_id]['cxr_text'] = pathlib.Path(
                    report_text_path).read_text()

        return all_data
