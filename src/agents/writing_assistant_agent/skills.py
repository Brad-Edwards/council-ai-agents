from council.skills import SkillBase
from council.contexts import ChatMessage, SkillContext, LLMContext
from council.llm import LLMBase, LLMMessage

from string import Template


class OutlineWriterSkill(SkillBase):
    """Write or revise the outline of an article."""

    def __init__(self, llm: LLMBase):
        """Build a new OutlineWriterSkill."""

        super().__init__(name="OutlineWriterSkill")

        self.llm = self.new_monitor("llm", llm)
        self.system_prompt = "You are an expert..."
        self.main_prompt_template = Template("# Task Description ...")

    def execute(self, context: SkillContext) -> ChatMessage:
        """Execute `OutlineWriterSkill`."""

        chat_message_history = [f"{m.kind}: {m.message}" for m in context.messages]

        article = context.last_message.data["article"]
        instructions = context.last_message.message
        outline = context.last_message.data["outline"]

        main_prompt = self.main_prompt_template.substitute(
            conversation_history=chat_message_history,
            article=article,
            article_outline=outline,
            instructions=instructions,
        )

        messages_to_llm = [
            LLMMessage.system_message(self.system_prompt),
            LLMMessage.assistant_message(main_prompt),
        ]

        llm_result = self.llm.inner.post_chat_request(
            context=LLMContext.from_context(context, self.llm),
            messages=messages_to_llm,
            temperature=0.1,
        )
        llm_response = llm_result.first_choice

        return ChatMessage.skill(
            source=self.name,
            message="I've edited the outline and placed the response in the 'data' field.",
            data={"outline": llm_response, "instructions": instructions},
        )
