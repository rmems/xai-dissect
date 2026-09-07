mod support;

use std::fs;

use xai_dissect::inventory::{InventoryConfig, build_inventory};
use xai_dissect::parser;
use xai_dissect::schema::{TensorDType, TensorKind, TensorRole};

use support::{PARSER_HEX_FIXTURE, decode_hex_fixture, unique_temp_root};

#[test]
fn parser_fixture_discovers_single_f32_tensor() {
    let root = unique_temp_root("parser-fixture");
    fs::create_dir_all(&root).expect("create temp dir");
    let shard = root.join("tensor0000.pkl");
    fs::write(&shard, decode_hex_fixture(PARSER_HEX_FIXTURE)).expect("write parser fixture");

    let tensors = parser::dissect_shard(&shard).expect("dissect shard");
    assert_eq!(tensors.len(), 1);
    assert_eq!(tensors[0].role, TensorRole::Tensor);
    assert_eq!(tensors[0].dtype, TensorDType::F32);
    assert_eq!(tensors[0].shape.render(), "(2, 4)");
    assert_eq!(tensors[0].nbytes, 32);

    let _ = fs::remove_dir_all(root);
}

#[test]
fn inventory_fixture_builds_without_real_weights() {
    let root = unique_temp_root("inventory-fixture");
    fs::create_dir_all(&root).expect("create temp dir");
    let shard = root.join("tensor0000.pkl");
    fs::write(&shard, decode_hex_fixture(PARSER_HEX_FIXTURE)).expect("write parser fixture");

    let inventory = build_inventory(
        &root,
        &InventoryConfig {
            prefix: "tensor".into(),
            limit: None,
            model_family: "grok-1".into(),
        },
    )
    .expect("build inventory");

    assert_eq!(inventory.shard_count, 1);
    assert_eq!(inventory.tensors.len(), 1);
    assert_eq!(inventory.inferred.vocab_size, Some(2));
    assert_eq!(inventory.inferred.d_model, Some(4));
    assert!(matches!(
        inventory.tensors[0].kind,
        TensorKind::TokenEmbedding
    ));

    let _ = fs::remove_dir_all(root);
}
