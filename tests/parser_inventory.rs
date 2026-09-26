mod support;

use std::fs;

use xai_dissect::inventory::{InventoryConfig, build_inventory, build_inventory_with_options};
use xai_dissect::parser;
use xai_dissect::schema::{TensorDType, TensorKind, TensorRole};

use support::{PARSER_HEX_FIXTURE, decode_hex_fixture, unique_temp_root};

fn write_single_shard_checkpoint(root: &std::path::Path, bytes: &[u8]) -> std::path::PathBuf {
    let shard = root.join("tensor0000.pkl");
    fs::write(&shard, bytes).expect("write shard");
    shard
}

fn corrupt_tensor_payload_length(bytes: &mut [u8]) {
    let payload_len = bytes
        .windows(2)
        .rposition(|window| window == [b'C', 32])
        .expect("fixture payload length");
    bytes[payload_len + 1] = 31;
}

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

#[test]
fn legacy_inventory_without_parser_diagnostics_deserializes_with_defaults() {
    let root = unique_temp_root("inventory-legacy-diagnostics");
    fs::create_dir_all(&root).expect("create temp dir");
    fs::write(
        root.join("tensor0000.pkl"),
        decode_hex_fixture(PARSER_HEX_FIXTURE),
    )
    .expect("write parser fixture");
    let inventory = build_inventory(&root, &InventoryConfig::default()).expect("build inventory");
    let mut json = serde_json::to_value(inventory).expect("serialize inventory");
    let object = json.as_object_mut().expect("inventory object");
    object.remove("skipped_anchors");
    object.remove("shard_parse_summaries");
    object.remove("skipped_anchor_count");

    let parsed: xai_dissect::schema::ModelInventory =
        serde_json::from_value(json).expect("deserialize legacy inventory");
    assert!(parsed.skipped_anchors.is_empty());
    assert!(parsed.shard_parse_summaries.is_empty());
    assert_eq!(parsed.skipped_anchor_count, 0);

    let _ = fs::remove_dir_all(root);
}

#[test]
fn malformed_anchor_is_reported_while_valid_tensor_is_retained() {
    let root = unique_temp_root("parser-skipped-anchor");
    fs::create_dir_all(&root).expect("create temp dir");
    let valid = decode_hex_fixture(PARSER_HEX_FIXTURE);
    let mut mixed = valid.clone();
    let mut malformed = valid;
    corrupt_tensor_payload_length(&mut malformed);
    mixed.extend_from_slice(&malformed);
    let shard = write_single_shard_checkpoint(&root, &mixed);

    let parsed = parser::dissect_shard_with_diagnostics(&shard).expect("dissect shard");
    assert_eq!(parsed.tensors.len(), 1);
    assert_eq!(parsed.skipped_anchors.len(), 1);
    assert!(
        parsed.skipped_anchors[0]
            .error
            .contains("shape/payload mismatch")
    );

    let config = InventoryConfig::default();
    let inventory = build_inventory(&root, &config).expect("build permissive inventory");
    assert_eq!(inventory.tensors.len(), 1);
    assert_eq!(inventory.skipped_anchor_count, 1);
    assert_eq!(inventory.skipped_anchors[0].shard_ordinal, 0);
    assert_eq!(inventory.skipped_anchors[0].shard_path, shard);
    assert_eq!(inventory.shard_parse_summaries[0].skipped_anchor_count, 1);

    let error = build_inventory_with_options(&root, &config, true)
        .expect_err("strict skipped-anchor policy must reject the inventory");
    assert!(
        error
            .to_string()
            .contains("skipped 1 malformed tensor anchor")
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn malformed_dtype_postamble_is_skipped_and_trips_strict_inventory() {
    let root = unique_temp_root("parser-bad-postamble");
    fs::create_dir_all(&root).expect("create temp dir");
    let mut bytes = decode_hex_fixture(PARSER_HEX_FIXTURE);
    const F32_DTYPE_TAG: &[u8] = b"\x8c\x02f4";
    let tag_pos = bytes
        .windows(F32_DTYPE_TAG.len())
        .position(|window| window == F32_DTYPE_TAG)
        .expect("f32 dtype tag");
    let postamble = tag_pos + F32_DTYPE_TAG.len();
    bytes[postamble] = 0xff;
    write_single_shard_checkpoint(&root, &bytes);

    let parsed = parser::dissect_shard_with_diagnostics(&root.join("tensor0000.pkl"))
        .expect("dissect shard");
    assert!(parsed.tensors.is_empty());
    assert_eq!(parsed.skipped_anchors.len(), 1);
    assert!(
        parsed.skipped_anchors[0]
            .error
            .contains("malformed dtype postamble")
    );

    let config = InventoryConfig::default();
    let inventory = build_inventory(&root, &config).expect("permissive inventory");
    assert_eq!(inventory.skipped_anchor_count, 1);

    build_inventory_with_options(&root, &config, true)
        .expect_err("strict mode must reject malformed postamble");

    let _ = fs::remove_dir_all(root);
}
