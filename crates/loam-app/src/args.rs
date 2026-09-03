use std::collections::HashMap;

#[derive(Clone, Debug, Default)]
pub struct Args {
    map: HashMap<String, String>,
    bare_flags: Vec<String>,
}

impl Args {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn current() -> Self {
        Self::from_argv(std::env::args().skip(1))
    }

    #[cfg(target_arch = "wasm32")]
    pub fn current() -> Self {
        let mut map = HashMap::new();
        if let Some(window) = web_sys::window() {
            if let Ok(search) = window.location().search() {
                parse_query_into(&search, &mut map);
            }
            if let Ok(hash) = window.location().hash() {
                parse_query_into(&hash, &mut map);
            }
        }
        QUERY_OVERRIDE.with(|q| {
            if let Some((search, hash)) = q.borrow().as_ref() {
                parse_query_into(search, &mut map);
                parse_query_into(hash, &mut map);
            }
        });
        Self {
            map,
            bare_flags: Vec::new(),
        }
    }

    /// Positionals are ignored; a bare `--key` is kept for [`Args::has_bare_flag`].
    pub fn from_argv<I, S>(argv: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut map = HashMap::new();
        let mut bare_flags = Vec::new();
        for arg in argv {
            let Some(stripped) = arg.as_ref().strip_prefix("--") else {
                continue;
            };
            match stripped.split_once('=') {
                Some((k, v)) if !k.is_empty() => {
                    map.insert(k.to_string(), v.to_string());
                }
                // A lone `--` is the end-of-flags marker, not a flag named "".
                None if !stripped.is_empty() => bare_flags.push(stripped.to_string()),
                _ => {}
            }
        }
        Self { map, bare_flags }
    }

    pub fn from_pairs<I, K, V>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        Self {
            map: pairs
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
            bare_flags: Vec::new(),
        }
    }

    /// Always false on wasm32: the query surface has no bare form.
    pub fn has_bare_flag(&self, key: &str) -> bool {
        self.bare_flags.iter().any(|flag| flag == key)
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.map.get(key).map(String::as_str)
    }

    pub fn parse<T: std::str::FromStr>(&self, key: &str) -> Option<T> {
        self.get(key)?.parse().ok()
    }

    /// Empty segments are filtered, so `?shapes=a,,b` yields `["a", "b"]`.
    pub fn get_many<'a>(&'a self, key: &str) -> Vec<&'a str> {
        match self.get(key) {
            Some(v) => v.split(',').filter(|s| !s.is_empty()).collect(),
            None => Vec::new(),
        }
    }

    /// Order is HashMap-arbitrary.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.map.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    // Workers have no `window.location`; the init message forwards the query.
    static QUERY_OVERRIDE: std::cell::RefCell<Option<(String, String)>> =
        const { std::cell::RefCell::new(None) };
}

/// Hash wins on collision, matching the main-thread parse order.
#[cfg(target_arch = "wasm32")]
pub fn set_query_override(search: String, hash: String) {
    QUERY_OVERRIDE.with(|q| *q.borrow_mut() = Some((search, hash)));
}

#[cfg(any(target_arch = "wasm32", test))]
fn parse_query_into(raw: &str, map: &mut HashMap<String, String>) {
    let trimmed = raw.trim_start_matches(['?', '#']);
    if trimmed.is_empty() {
        return;
    }
    for pair in trimmed.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            if !k.is_empty() {
                let value = v.replace('+', " ");
                map.insert(k.to_string(), value);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_returns_typed_or_none() {
        let args = Args::from_pairs([("seed", "42"), ("fov", "60.5"), ("bad", "nope")]);
        assert_eq!(args.parse::<u32>("seed"), Some(42));
        assert_eq!(args.parse::<f32>("fov"), Some(60.5));
        assert_eq!(args.parse::<u32>("bad"), None);
        assert_eq!(args.parse::<u32>("missing"), None);
    }

    #[test]
    fn from_argv_keeps_only_attached_values_and_ignores_positionals() {
        let args = Args::from_argv(["--seed=42", "sub", "--", "--=x", "--fov=60.5"]);
        assert_eq!(args.get("seed"), Some("42"));
        assert_eq!(args.get("fov"), Some("60.5"));
        assert_eq!(args.get("sub"), None);
        assert_eq!(args.get(""), None);
    }

    #[test]
    fn argv_pairs_split_on_the_first_equals_so_values_keep_the_rest() {
        let args = Args::from_argv(["--state=a=b", "--a=1=2=3", "--eq==", "--t=abc=="]);
        assert_eq!(args.get("state"), Some("a=b"));
        assert_eq!(args.get("a"), Some("1=2=3"));
        assert_eq!(args.get("eq"), Some("="));
        assert_eq!(args.get("t"), Some("abc=="));

        assert_eq!(args.get("state=a"), None);
        assert_eq!(args.get("eq="), None);
        assert!(!args.has_bare_flag("state=a=b"));
    }

    #[test]
    fn a_bare_flag_is_recorded_and_its_value_is_not_absorbed() {
        let args = Args::from_argv(["--shapes", "5-cell,8-cell", "--seed=42"]);
        assert_eq!(args.get("shapes"), None);
        assert!(args.has_bare_flag("shapes"));

        assert!(!args.has_bare_flag("seed"));
        assert!(!args.has_bare_flag("5-cell,8-cell"));
        assert!(!args.has_bare_flag(""));
        assert!(!Args::from_pairs([("shapes", "5-cell")]).has_bare_flag("shapes"));
    }

    #[test]
    fn get_many_splits_on_comma_and_drops_empties() {
        let args = Args::from_pairs([("shapes", "tesseract,5-cell,,8-cell,")]);
        assert_eq!(
            args.get_many("shapes"),
            vec!["tesseract", "5-cell", "8-cell"]
        );
        assert!(args.get_many("missing").is_empty());
    }

    fn parse_all(fragments: &[&str]) -> Vec<(String, String)> {
        let mut map = HashMap::new();
        for fragment in fragments {
            parse_query_into(fragment, &mut map);
        }
        let mut pairs: Vec<(String, String)> = map.into_iter().collect();
        pairs.sort();
        pairs
    }

    fn pairs(expected: &[(&str, &str)]) -> Vec<(String, String)> {
        expected
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn query_leading_markers_are_stripped_regardless_of_count_or_kind() {
        assert_eq!(parse_all(&["?a=1&b=2"]), pairs(&[("a", "1"), ("b", "2")]));
        assert_eq!(parse_all(&["#a=1"]), pairs(&[("a", "1")]));
        assert_eq!(parse_all(&["??a=1"]), pairs(&[("a", "1")]));
        assert_eq!(parse_all(&["#?a=1"]), pairs(&[("a", "1")]));
    }

    #[test]
    fn query_segments_without_a_nonempty_key_and_separator_are_dropped() {
        assert_eq!(parse_all(&["?flag"]), pairs(&[]));
        assert_eq!(parse_all(&["?=1"]), pairs(&[]));
        assert_eq!(parse_all(&["?"]), pairs(&[]));
        assert_eq!(parse_all(&[""]), pairs(&[]));

        assert_eq!(parse_all(&["?flag&=1&a=2"]), pairs(&[("a", "2")]));
    }

    #[test]
    fn query_values_decode_plus_as_space_and_keep_empty_values() {
        assert_eq!(
            parse_all(&["?title=hello+world&plus=a+b+c&empty="]),
            pairs(&[("empty", ""), ("plus", "a b c"), ("title", "hello world")])
        );
    }

    #[test]
    fn query_repeated_keys_resolve_last_write_wins_within_and_across_fragments() {
        assert_eq!(parse_all(&["?a=1&a=2&a=3"]), pairs(&[("a", "3")]));

        assert_eq!(
            parse_all(&["?a=search&b=only-search", "#a=hash"]),
            pairs(&[("a", "hash"), ("b", "only-search")])
        );
    }

    #[test]
    fn query_pairs_split_on_the_first_equals_so_values_keep_the_rest() {
        assert_eq!(parse_all(&["?state=a=b"]), pairs(&[("state", "a=b")]));
        assert_eq!(parse_all(&["?a=1=2=3"]), pairs(&[("a", "1=2=3")]));
        assert_eq!(parse_all(&["?eq=="]), pairs(&[("eq", "=")]));

        assert_eq!(parse_all(&["?t=abc=="]), pairs(&[("t", "abc==")]));
        assert_eq!(parse_all(&["?t=YQ="]), pairs(&[("t", "YQ=")]));

        assert_eq!(
            parse_all(&["?a=x=y&b=2"]),
            pairs(&[("a", "x=y"), ("b", "2")])
        );
    }

    #[test]
    fn query_empty_fragment_leaves_prior_entries_intact() {
        assert_eq!(parse_all(&["?a=1", "#", ""]), pairs(&[("a", "1")]));
    }
}
