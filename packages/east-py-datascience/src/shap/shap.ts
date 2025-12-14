/**
 * Copyright (c) 2025 Elara AI Pty Ltd
 * Dual-licensed under AGPL-3.0 and commercial license. See LICENSE for details.
 */

/**
 * SHAP platform functions for East.
 *
 * Provides model-agnostic feature importance and explainability using SHAP values.
 * Uses cloudpickle for explainer serialization.
 *
 * @packageDocumentation
 */

import {
    East,
    StructType,
    VariantType,
    OptionType,
    IntegerType,
    FloatType,
    StringType,
    ArrayType,
    BlobType,
    NullType,
} from "@elaraai/east";
import { VectorType, MatrixType } from "../types.js";

// Re-export shared types for convenience
export { VectorType, MatrixType } from "../types.js";

// ============================================================================
// Data Types
// ============================================================================

/** String vector type for feature names */
export const StringVectorType = ArrayType(StringType);

// ============================================================================
// Result Types
// ============================================================================

/**
 * SHAP values type - variant for 2D (regression/binary) or 3D (multi-class).
 */
export const ShapValuesType = VariantType({
    /** 2D matrix for regression or binary classification (n_samples x n_features) */
    matrix_2d: MatrixType,
    /** 3D tensor for multi-class classification (n_samples x n_features x n_classes) */
    tensor_3d: ArrayType(MatrixType),
});

/**
 * Base value type - variant for single (regression/binary) or per-class (multi-class).
 */
export const ShapBaseValueType = VariantType({
    /** Single base value for regression or binary classification */
    single: FloatType,
    /** Per-class base values for multi-class classification */
    per_class: VectorType,
});

/**
 * Result type for SHAP value computation.
 */
export const ShapResultType = StructType({
    /** SHAP values - 2D matrix or 3D tensor depending on model type */
    shap_values: ShapValuesType,
    /** Base value(s) - single float or per-class array */
    base_value: ShapBaseValueType,
    /** Feature names */
    feature_names: StringVectorType,
});

/**
 * Result type for feature importance.
 */
export const FeatureImportanceType = StructType({
    /** Feature names */
    feature_names: StringVectorType,
    /** Mean absolute SHAP value for each feature */
    importances: VectorType,
    /** Standard deviation of absolute SHAP values */
    std: OptionType(VectorType),
});

// ============================================================================
// Model Blob Types
// ============================================================================

/**
 * Model blob type for serialized SHAP explainers.
 */
export const ShapModelBlobType = VariantType({
    /** SHAP TreeExplainer for tree-based models */
    shap_tree_explainer: StructType({
        /** Cloudpickle serialized explainer */
        data: BlobType,
        /** Number of input features */
        n_features: IntegerType,
    }),
    /** SHAP KernelExplainer for any model */
    shap_kernel_explainer: StructType({
        /** Cloudpickle serialized explainer */
        data: BlobType,
        /** Number of input features */
        n_features: IntegerType,
    }),
});

/**
 * Tree-based model blob type - accepts XGBoost and LightGBM models.
 */
export const TreeModelBlobType = VariantType({
    /** XGBoost regressor */
    xgboost_regressor: StructType({
        data: BlobType,
        n_features: IntegerType,
    }),
    /** XGBoost classifier */
    xgboost_classifier: StructType({
        data: BlobType,
        n_features: IntegerType,
        n_classes: IntegerType,
    }),
    /** XGBoost quantile regressor (uses median quantile for explanations) */
    xgboost_quantile: StructType({
        data: BlobType,
        quantiles: VectorType,
        n_features: IntegerType,
    }),
    /** LightGBM regressor */
    lightgbm_regressor: StructType({
        data: BlobType,
        n_features: IntegerType,
    }),
    /** LightGBM classifier */
    lightgbm_classifier: StructType({
        data: BlobType,
        n_features: IntegerType,
        n_classes: IntegerType,
    }),
});

/**
 * Any model blob type - accepts any model for kernel explainer.
 * Includes all tree-based models plus NGBoost, GP, and Torch.
 */
export const AnyModelBlobType = VariantType({
    // Tree-based
    xgboost_regressor: StructType({
        data: BlobType,
        n_features: IntegerType,
    }),
    xgboost_classifier: StructType({
        data: BlobType,
        n_features: IntegerType,
        n_classes: IntegerType,
    }),
    xgboost_quantile: StructType({
        data: BlobType,
        quantiles: VectorType,
        n_features: IntegerType,
    }),
    lightgbm_regressor: StructType({
        data: BlobType,
        n_features: IntegerType,
    }),
    lightgbm_classifier: StructType({
        data: BlobType,
        n_features: IntegerType,
        n_classes: IntegerType,
    }),
    // NGBoost
    ngboost_regressor: StructType({
        data: BlobType,
        distribution: VariantType({
            normal: NullType,
            lognormal: NullType,
        }),
        n_features: IntegerType,
    }),
    // GP
    gp_regressor: StructType({
        data: BlobType,
        n_features: IntegerType,
        kernel_type: StringType,
    }),
    // Torch
    torch_mlp: StructType({
        data: BlobType,
        n_features: IntegerType,
        hidden_layers: ArrayType(IntegerType),
        output_dim: IntegerType,
    }),
});

// ============================================================================
// Platform Functions
// ============================================================================

/**
 * Create a SHAP TreeExplainer for tree-based models.
 *
 * Works with XGBoost and LightGBM models (regressor and classifier).
 *
 * @param model - Tree-based model blob (XGBoost or LightGBM)
 * @returns SHAP TreeExplainer blob
 */
export const shap_tree_explainer_create = East.platform(
    "shap_tree_explainer_create",
    [TreeModelBlobType],
    ShapModelBlobType
);

/**
 * Create a SHAP KernelExplainer for any model.
 *
 * Works with any model that has a predict method (NGBoost, GP, Torch, etc.).
 * Requires background data for computing expected values.
 *
 * @param model - Any model blob
 * @param X_background - Background data for computing expected values
 * @returns SHAP KernelExplainer blob
 */
export const shap_kernel_explainer_create = East.platform(
    "shap_kernel_explainer_create",
    [AnyModelBlobType, MatrixType],
    ShapModelBlobType
);

/**
 * Compute SHAP values for samples.
 *
 * @param explainer - SHAP explainer blob
 * @param X - Feature matrix to explain
 * @param feature_names - Names of features
 * @returns SHAP values, base value, and feature names
 */
export const shap_compute_values = East.platform(
    "shap_compute_values",
    [ShapModelBlobType, MatrixType, StringVectorType],
    ShapResultType
);

/**
 * Compute global feature importance from SHAP values.
 *
 * @param shap_values - SHAP values (2D matrix or 3D tensor)
 * @param feature_names - Names of features
 * @returns Feature importance with mean |SHAP| values
 */
export const shap_feature_importance = East.platform(
    "shap_feature_importance",
    [ShapValuesType, StringVectorType],
    FeatureImportanceType
);

// ============================================================================
// Grouped Export
// ============================================================================

/**
 * Type definitions for SHAP functions.
 */
export const ShapTypes = {
    /** Vector type (array of floats) */
    VectorType,
    /** Matrix type (2D array of floats) */
    MatrixType,
    /** String vector type */
    StringVectorType,
    /** SHAP values variant type (2D or 3D) */
    ShapValuesType,
    /** SHAP base value variant type (single or per-class) */
    ShapBaseValueType,
    /** SHAP result type */
    ShapResultType,
    /** Feature importance type */
    FeatureImportanceType,
    /** SHAP explainer model blob type */
    ShapModelBlobType,
    /** Tree model blob type for input */
    TreeModelBlobType,
    /** Any model blob type for kernel explainer */
    AnyModelBlobType,
} as const;

/**
 * SHAP explainability functions.
 *
 * Provides model-agnostic feature importance and SHAP value computation.
 *
 * @example
 * ```ts
 * import { East, variant } from "@elaraai/east";
 * import { Shap, LightGBM } from "@elaraai/east-py-datascience";
 *
 * const explain = East.function([LightGBM.Types.ModelBlobType, Shap.Types.MatrixType], Shap.Types.ShapResultType, ($, model, X) => {
 *     // Create explainer
 *     const explainer = $.let(Shap.treeExplainerCreate(model));
 *
 *     // Compute SHAP values
 *     const feature_names = $.let(["feature1", "feature2"]);
 *     const result = $.let(Shap.computeValues(explainer, X, feature_names));
 *
 *     return $.return(result);
 * });
 * ```
 */
export const Shap = {
    /** Create TreeExplainer for tree-based models */
    treeExplainerCreate: shap_tree_explainer_create,
    /** Create KernelExplainer for any model */
    kernelExplainerCreate: shap_kernel_explainer_create,
    /** Compute SHAP values */
    computeValues: shap_compute_values,
    /** Compute feature importance from SHAP values */
    featureImportance: shap_feature_importance,
    /** Type definitions */
    Types: ShapTypes,
} as const;
