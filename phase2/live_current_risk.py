from pathlib import Path
import math

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "umn_zone_abnormal_classifier.joblib"
)


FEATURE_COLUMNS = [
    "zone_presence_ratio",
    "unique_person_count",
    "avg_person_count_per_frame",
    "max_person_count_per_frame",
    "avg_bbox_area_ratio",
    "max_bbox_area_ratio",
    "avg_detection_confidence",
    "valid_motion_rows",
    "avg_movement_distance_norm",
    "avg_speed_norm_s",
    "max_speed_norm_s",
    "speed_std_norm_s",
    "avg_acceleration_norm_s2",
    "acceleration_std_norm_s2",
    "avg_direction_change_deg",
    "direction_consistency",
    "movement_activity_ratio",
]


class LiveCurrentRiskPredictor:

    def __init__(
        self,
        fps,
        frame_width,
        frame_height,
        window_seconds=2.0,
    ):

        self.fps = fps

        self.frame_width = frame_width
        self.frame_height = frame_height

        self.window_seconds = window_seconds

        self.window_frames = max(
            1,
            int(
                round(
                    fps * window_seconds
                )
            )
        )

        self.frame_diagonal = math.sqrt(
            frame_width ** 2
            + frame_height ** 2
        )

        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                f"Current risk model not found:\n{MODEL_PATH}"
            )

        print()
        print(
            "Loading Current Risk Model..."
        )

        self.model_package = joblib.load(
            MODEL_PATH
        )

        if isinstance(
            self.model_package,
            dict
        ):

            if "model" in self.model_package:

                self.model = self.model_package[
                    "model"
                ]

            elif "pipeline" in self.model_package:

                self.model = self.model_package[
                    "pipeline"
                ]

            else:

                self.model = self.model_package

        else:

            self.model = self.model_package


        self.reset_window()


    def reset_window(self):

        self.window_start_frame = None

        self.zone_rows = {
            "ZONE_A": [],
            "ZONE_B": [],
            "ZONE_C": [],
        }

        self.zone_frame_counts = {
            "ZONE_A": [],
            "ZONE_B": [],
            "ZONE_C": [],
        }


    def add_frame(
        self,
        frame_number,
        zone_people,
    ):

        if self.window_start_frame is None:

            self.window_start_frame = frame_number


        for zone_id in [
            "ZONE_A",
            "ZONE_B",
            "ZONE_C",
        ]:

            people = zone_people.get(
                zone_id,
                []
            )

            self.zone_frame_counts[
                zone_id
            ].append(
                len(
                    people
                )
            )

            for person in people:

                self.zone_rows[
                    zone_id
                ].append(
                    person
                )


        frames_in_window = (
            frame_number
            - self.window_start_frame
            + 1
        )


        if frames_in_window < self.window_frames:

            return None


        predictions = (
            self.predict_current_window()
        )

        self.reset_window()

        return predictions


    def predict_current_window(self):

        predictions = {}

        for zone_id in [
            "ZONE_A",
            "ZONE_B",
            "ZONE_C",
        ]:

            feature_row = (
                self.build_zone_features(
                    zone_id
                )
            )


            dataframe = pd.DataFrame(
                [
                    feature_row
                ],
                columns=FEATURE_COLUMNS,
            )


            probability = float(
                self.model.predict_proba(
                    dataframe
                )[0][1]
            )


            prediction_value = int(
                self.model.predict(
                    dataframe
                )[0]
            )


            prediction_label = (
                "ABNORMAL"
                if prediction_value == 1
                else "NORMAL"
            )


            predictions[
                zone_id
            ] = {
                "label": prediction_label,
                "probability": probability,
                "features": feature_row,
            }


        return predictions


    def build_zone_features(
        self,
        zone_id,
    ):

        rows = self.zone_rows[
            zone_id
        ]

        frame_counts = (
            self.zone_frame_counts[
                zone_id
            ]
        )


        expected_frames = (
            self.window_frames
        )


        observed_frames = sum(
            1
            for count in frame_counts
            if count > 0
        )


        zone_presence_ratio = (
            observed_frames
            / expected_frames
            if expected_frames > 0
            else 0.0
        )


        if not rows:

            return {
                column: 0.0
                for column
                in FEATURE_COLUMNS
            }


        track_ids = [
            row["track_id"]
            for row in rows
            if row["track_id"] is not None
        ]


        unique_person_count = len(
            set(
                track_ids
            )
        )


        avg_person_count = float(
            np.mean(
                frame_counts
            )
        )


        max_person_count = float(
            np.max(
                frame_counts
            )
        )


        bbox_area = np.array(
            [
                row["bbox_area_ratio"]
                for row in rows
            ],
            dtype=float,
        )


        detection_conf = np.array(
            [
                row["detection_confidence"]
                for row in rows
            ],
            dtype=float,
        )


        motion_valid = np.array(
            [
                row["motion_valid"]
                for row in rows
            ],
            dtype=float,
        )


        movement_norm = np.array(
            [
                row["movement_distance_norm"]
                for row in rows
            ],
            dtype=float,
        )


        speed_norm = np.array(
            [
                row["speed_norm_s"]
                for row in rows
            ],
            dtype=float,
        )


        acceleration_norm = np.array(
            [
                row["acceleration_norm_s2"]
                for row in rows
            ],
            dtype=float,
        )


        direction_change = np.array(
            [
                row["direction_change_deg"]
                for row in rows
            ],
            dtype=float,
        )


        direction_sin = np.array(
            [
                row["direction_sin"]
                for row in rows
            ],
            dtype=float,
        )


        direction_cos = np.array(
            [
                row["direction_cos"]
                for row in rows
            ],
            dtype=float,
        )


        mean_sin = float(
            np.mean(
                direction_sin
            )
        )


        mean_cos = float(
            np.mean(
                direction_cos
            )
        )


        direction_consistency = float(
            math.sqrt(
                mean_sin ** 2
                + mean_cos ** 2
            )
        )


        movement_activity_ratio = float(
            np.mean(
                speed_norm > 0
            )
        )


        return {

            "zone_presence_ratio":
                float(
                    zone_presence_ratio
                ),

            "unique_person_count":
                float(
                    unique_person_count
                ),

            "avg_person_count_per_frame":
                avg_person_count,

            "max_person_count_per_frame":
                max_person_count,

            "avg_bbox_area_ratio":
                float(
                    np.mean(
                        bbox_area
                    )
                ),

            "max_bbox_area_ratio":
                float(
                    np.max(
                        bbox_area
                    )
                ),

            "avg_detection_confidence":
                float(
                    np.mean(
                        detection_conf
                    )
                ),

            "valid_motion_rows":
                float(
                    np.sum(
                        motion_valid
                    )
                ),

            "avg_movement_distance_norm":
                float(
                    np.mean(
                        movement_norm
                    )
                ),

            "avg_speed_norm_s":
                float(
                    np.mean(
                        speed_norm
                    )
                ),

            "max_speed_norm_s":
                float(
                    np.max(
                        speed_norm
                    )
                ),

            "speed_std_norm_s":
                float(
                    np.std(
                        speed_norm
                    )
                ),

            "avg_acceleration_norm_s2":
                float(
                    np.mean(
                        acceleration_norm
                    )
                ),

            "acceleration_std_norm_s2":
                float(
                    np.std(
                        acceleration_norm
                    )
                ),

            "avg_direction_change_deg":
                float(
                    np.mean(
                        direction_change
                    )
                ),

            "direction_consistency":
                direction_consistency,

            "movement_activity_ratio":
                movement_activity_ratio,
        }