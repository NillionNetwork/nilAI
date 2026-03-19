use nilai_domain::chat::Message;
use nilai_domain::error::{NilaiError, NilaiResult};
use nilai_domain::ports::SearchProvider;
use nilai_domain::search::Source;

/// Result of web search augmentation.
pub struct WebSearchEnhancedMessages {
    pub messages: Vec<Message>,
    pub sources: Vec<Source>,
}

/// Augment chat messages with web search results.
///
/// Takes the user's last message as a search query, searches the web,
/// and prepends a system message with the search results.
pub async fn augment_with_search(
    messages: Vec<Message>,
    search_provider: &dyn SearchProvider,
    search_count: u32,
) -> NilaiResult<WebSearchEnhancedMessages> {
    // Extract the last user message as search query
    let query = messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .and_then(|m| match &m.content {
            Some(nilai_domain::chat::MessageContent::Text(text)) => Some(text.clone()),
            Some(nilai_domain::chat::MessageContent::Parts(parts)) => {
                parts.iter().find_map(|p| match p {
                    nilai_domain::chat::ContentPart::Text { text } => Some(text.clone()),
                    _ => None,
                })
            }
            None => None,
        })
        .ok_or_else(|| {
            NilaiError::BadRequest("No user message found for web search".to_string())
        })?;

    // Search
    let results = search_provider.search(&query, search_count).await?;

    if results.is_empty() {
        return Ok(WebSearchEnhancedMessages {
            messages,
            sources: vec![],
        });
    }

    // Build sources
    let sources: Vec<Source> = results
        .iter()
        .map(|r| Source {
            source: r.url.clone(),
            content: r.body.clone(),
        })
        .collect();

    // Format search context as system message
    let mut context_parts = vec![
        "The following information was retrieved from web search results. Use this context to inform your response:\n".to_string(),
    ];

    for (i, result) in results.iter().enumerate() {
        context_parts.push(format!(
            "Source {}: {} ({})\n{}\n",
            i + 1,
            result.title,
            result.url,
            result.body,
        ));
    }

    let search_context = context_parts.join("\n");

    // Prepend system message with search context
    let search_message = Message {
        role: "system".to_string(),
        content: Some(nilai_domain::chat::MessageContent::Text(search_context)),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    };

    let mut enhanced_messages = vec![search_message];
    enhanced_messages.extend(messages);

    Ok(WebSearchEnhancedMessages {
        messages: enhanced_messages,
        sources,
    })
}
