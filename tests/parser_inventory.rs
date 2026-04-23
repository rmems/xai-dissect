mod support;

use std::fs;
use std::path::Path;

use xai_dissect::inventory::{InventoryConfig, build_inventory};
use xai_dissect::parser;
use xai_dissect::schema::{TensorDType, TensorKind, TensorRole};

use support::unique_temp_root;

#[test]
fn parser_fixture_discovers_single_f32_tensor() {
    let root = unique_temp_root("parser-fixture");
    fs::create_dir_all(&root).expect("create temp dir");
    let shard = root.join("tensor0000.pkl");
    fs::write(
        &shard,
        decode_hex_fixture("tests/fixtures/parser/single_f32_tensor.pkl.hex"),
    )
    .expect("write parser fixture");

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
    fs::write(
        &shard,
        decode_hex_fixture("tests/fixtures/parser/single_f32_tensor.pkl.hex"),
    )
    .expect("write parser fixture");

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

fn decode_hex_fixture(rel_path: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel_path);
    let hex = fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("read fixture {}: {err}", path.display()));
    let hex = hex
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    assert_eq!(hex.len() % 2, 0, "fixture must have an even hex length");

    let mut out = Vec::with_capacity(hex.len() / 2);
    let mut i = 0;
    while i < hex.len() {
        let byte = u8::from_str_radix(&hex[i..i + 2], 16).expect("hex byte");
        out.push(byte);
        i += 2;
    }
    out
}
