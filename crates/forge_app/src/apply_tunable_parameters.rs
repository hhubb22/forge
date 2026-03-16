use forge_domain::{Agent, Conversation, ToolDefinition};

/// Applies tunable parameters from agent to conversation context
#[derive(Debug, Clone)]
pub struct ApplyTunableParameters {
    agent: Agent,
    tool_definitions: Vec<ToolDefinition>,
    reasoning_override: Option<forge_domain::ReasoningConfig>,
    reasoning_supported: bool,
}

impl ApplyTunableParameters {
    pub const fn new(agent: Agent, tool_definitions: Vec<ToolDefinition>) -> Self {
        Self {
            agent,
            tool_definitions,
            reasoning_override: None,
            reasoning_supported: true,
        }
    }

    pub(crate) fn reasoning_override(
        mut self,
        reasoning_override: Option<forge_domain::ReasoningConfig>,
    ) -> Self {
        self.reasoning_override = reasoning_override;
        self
    }

    pub(crate) const fn reasoning_supported(mut self, reasoning_supported: bool) -> Self {
        self.reasoning_supported = reasoning_supported;
        self
    }

    pub fn apply(self, mut conversation: Conversation) -> Conversation {
        let mut ctx = conversation.context.take().unwrap_or_default();

        if let Some(temperature) = self.agent.temperature {
            ctx = ctx.temperature(temperature);
        }
        if let Some(top_p) = self.agent.top_p {
            ctx = ctx.top_p(top_p);
        }
        if let Some(top_k) = self.agent.top_k {
            ctx = ctx.top_k(top_k);
        }
        if let Some(max_tokens) = self.agent.max_tokens {
            ctx = ctx.max_tokens(max_tokens.value() as usize);
        }
        if self.reasoning_supported {
            if let Some(ref reasoning) = self.reasoning_override {
                ctx = ctx.reasoning(reasoning.clone());
            } else if let Some(ref reasoning) = self.agent.reasoning {
                ctx = ctx.reasoning(reasoning.clone());
            }
        } else {
            ctx.reasoning = None;
        }

        conversation.context(ctx.tools(self.tool_definitions))
    }
}

#[cfg(test)]
mod tests {
    use forge_domain::{
        AgentId, Context, ConversationId, MaxTokens, ModelId, ProviderId, ReasoningConfig,
        Temperature, ToolDefinition, TopK, TopP,
    };
    use pretty_assertions::assert_eq;

    use super::*;

    #[derive(schemars::JsonSchema)]
    struct TestToolInput;

    #[test]
    fn test_apply_sets_parameters() {
        let reasoning = ReasoningConfig::default().max_tokens(2000);

        let agent = Agent::new(
            AgentId::new("test"),
            ProviderId::ANTHROPIC,
            ModelId::new("claude-3-5-sonnet-20241022"),
        )
        .temperature(Temperature::new(0.7).unwrap())
        .max_tokens(MaxTokens::new(1000).unwrap())
        .top_k(TopK::new(50).unwrap())
        .top_p(TopP::new(0.9).unwrap())
        .reasoning(reasoning.clone());

        let tool_def = ToolDefinition::new("test_tool")
            .description("A test tool")
            .input_schema(schemars::schema_for!(TestToolInput));

        let conversation =
            Conversation::new(ConversationId::generate()).context(Context::default());

        let actual = ApplyTunableParameters::new(agent, vec![tool_def.clone()]).apply(conversation);

        let ctx = actual.context.unwrap();
        assert_eq!(ctx.temperature, Some(Temperature::new(0.7).unwrap()));
        assert_eq!(ctx.max_tokens, Some(1000));
        assert_eq!(ctx.top_k, Some(TopK::new(50).unwrap()));
        assert_eq!(ctx.top_p, Some(TopP::new(0.9).unwrap()));
        assert_eq!(ctx.reasoning, Some(reasoning));
        assert_eq!(ctx.tools, vec![tool_def]);
    }

    #[test]
    fn test_apply_uses_reasoning_override_over_agent_reasoning() {
        let fixture = Agent::new(
            AgentId::new("test"),
            ProviderId::ANTHROPIC,
            ModelId::new("claude-3-5-sonnet-20241022"),
        )
        .reasoning(ReasoningConfig::default().effort(forge_domain::Effort::Low));
        let override_reasoning = ReasoningConfig::default()
            .enabled(true)
            .effort(forge_domain::Effort::High);
        let conversation =
            Conversation::new(ConversationId::generate()).context(Context::default());

        let actual = ApplyTunableParameters::new(fixture, vec![])
            .reasoning_override(Some(override_reasoning.clone()))
            .apply(conversation);
        let expected = Some(override_reasoning);

        assert_eq!(actual.context.unwrap().reasoning, expected);
    }

    #[test]
    fn test_apply_drops_reasoning_when_model_does_not_support_it() {
        let fixture = Agent::new(
            AgentId::new("test"),
            ProviderId::ANTHROPIC,
            ModelId::new("claude-3-5-sonnet-20241022"),
        )
        .reasoning(ReasoningConfig::default().effort(forge_domain::Effort::High));
        let conversation =
            Conversation::new(ConversationId::generate()).context(Context::default());

        let actual = ApplyTunableParameters::new(fixture, vec![])
            .reasoning_supported(false)
            .apply(conversation);
        let expected = None;

        assert_eq!(actual.context.unwrap().reasoning, expected);
    }
}
