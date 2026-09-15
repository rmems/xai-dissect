// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//! Model-family adapters: layout constants, slot specs, and coverage
//! validators that sit beside generic inventory rather than inside it.
//!
//! Inventory entry points (`build_inventory`, classification, shard walk)
//! stay family-agnostic. Grok-1 complete-manifest coverage is invoked
//! explicitly through [`grok1`].

pub mod grok1;
