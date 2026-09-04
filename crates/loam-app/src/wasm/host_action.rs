//! The OffscreenCanvas-in-Worker architecture has no DOM access, so browser
//! APIs that need it (Pointer Lock, Fullscreen, Clipboard) must round-trip to
//! the main thread.

use std::cell::RefCell;

use wasm_bindgen::JsValue;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostAction {
    /// The transition is confirmed asynchronously via
    /// `InputMessage::PointerLockChanged(true)`.
    PointerLockRequest,
    /// The browser also auto-releases on Esc or tab switch; both paths
    /// round-trip back via `PointerLockChanged`.
    PointerLockRelease,
}

impl HostAction {
    // Source of truth for the wire format; `main_launcher`'s dispatch matches
    // the same literals.
    const fn kind_str(self) -> &'static str {
        match self {
            HostAction::PointerLockRequest => "pointer_lock_request",
            HostAction::PointerLockRelease => "pointer_lock_release",
        }
    }
}

thread_local! {
    static PENDING: RefCell<Vec<HostAction>> = const { RefCell::new(Vec::new()) };
}

pub fn queue(action: HostAction) {
    PENDING.with(|p| p.borrow_mut().push(action));
}

/// Posts one `{kind: "host_action", actions: [...]}` message; no-op if empty.
pub fn post_pending_actions(scope: &web_sys::DedicatedWorkerGlobalScope) -> anyhow::Result<()> {
    let actions = PENDING.with(|p| std::mem::take(&mut *p.borrow_mut()));
    if actions.is_empty() {
        return Ok(());
    }
    let msg = encode_actions(&actions)?;
    scope
        .post_message(&msg)
        .map_err(|e| anyhow::anyhow!("post host_action: {e:?}"))
}

fn encode_actions(actions: &[HostAction]) -> anyhow::Result<JsValue> {
    let msg = js_sys::Object::new();
    js_sys::Reflect::set(
        &msg,
        &JsValue::from_str("kind"),
        &JsValue::from_str("host_action"),
    )
    .map_err(|e| anyhow::anyhow!("set kind: {e:?}"))?;
    let arr = js_sys::Array::new();
    for action in actions {
        let item = js_sys::Object::new();
        js_sys::Reflect::set(
            &item,
            &JsValue::from_str("kind"),
            &JsValue::from_str(action.kind_str()),
        )
        .map_err(|e| anyhow::anyhow!("set action kind: {e:?}"))?;
        arr.push(&item);
    }
    js_sys::Reflect::set(&msg, &JsValue::from_str("actions"), &arr)
        .map_err(|e| anyhow::anyhow!("set actions: {e:?}"))?;
    Ok(msg.into())
}
