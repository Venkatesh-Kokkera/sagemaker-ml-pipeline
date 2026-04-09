import boto3
import sagemaker
import argparse
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    TransformStep
)
from sagemaker.workflow.parameters import (
    ParameterString,
    ParameterFloat,
    ParameterInteger
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.xgboost import XGBoost
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet

def get_session(region: str, role_arn: str):
    """Create SageMaker session."""
    boto_session = boto3.Session(region_name=region)
    sagemaker_session = sagemaker.Session(
        boto_session=boto_session
    )
    return sagemaker_session

def create_pipeline(args):
    """Create and deploy SageMaker pipeline."""
    print(f"Creating pipeline: {args.pipeline_name}")

    session = get_session(args.region, args.role_arn)

    # Pipeline parameters
    input_data = ParameterString(
        name="InputData",
        default_value=f"s3://{args.s3_bucket}/data/raw/"
    )
    accuracy_threshold = ParameterFloat(
        name="AccuracyThreshold",
        default_value=0.85
    )
    training_epochs = ParameterInteger(
        name="TrainingEpochs",
        default_value=100
    )

    # Step 1: Data Processing
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type="ml.m5.large",
        instance_count=1,
        role=args.role_arn,
        sagemaker_session=session
    )

    processing_step = ProcessingStep(
        name="DataProcessing",
        processor=sklearn_processor,
        inputs=[
            sagemaker.processing.ProcessingInput(
                source=input_data,
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            sagemaker.processing.ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/train",
                destination=f"s3://{args.s3_bucket}/data/processed/train/"
            ),
            sagemaker.processing.ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/test",
                destination=f"s3://{args.s3_bucket}/data/processed/test/"
            )
        ],
        code="scripts/feature_engineering.py"
    )

    # Step 2: Model Training
    xgb_estimator = XGBoost(
        entry_point="pipeline/steps/training.py",
        framework_version="1.7-1",
        instance_type="ml.m5.xlarge",
        instance_count=1,
        role=args.role_arn,
        sagemaker_session=session,
        hyperparameters={
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8
        }
    )

    training_step = TrainingStep(
        name="ModelTraining",
        estimator=xgb_estimator,
        inputs={
            "train": sagemaker.inputs.TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv"
