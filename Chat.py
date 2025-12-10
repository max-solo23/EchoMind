class Chat:
    def __init__(self, person, llm, llm_model, llm_tools, evaluator_llm):
        self.llm = llm
        self.llm_model = llm_model
        self.llm_tools = llm_tools
        self.person = person
        self.evaluator_llm = evaluator_llm

    def chat(self, message, history):
        person_system_prompt = self.person.system_prompt

        messages = [
            {"role": "system", "content": person_system_prompt}
                           ] + history + [
            {"role": "user", "content": message}
        ]

        done = False
        while not done:
            response = self.llm.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                tools=self.llm_tools.tools
            )

            finish_reason = response.choices[0].finish_reason
            msg = response.choices[0].message

            if finish_reason == "tool_calls":
                tool_calls = msg.tool_calls
                results = self.llm_tools.handle_tool_call(tool_calls)
                messages.append(msg)
                messages.extend(results)
            else:
                done = True

        reply = msg.content
        evaluation = self.evaluator_llm.evaluate(reply, message, history)

        if evaluation.is_acceptable:
            print("Passed evaluation - returning reply")
        else:
            print("Failed evaluation - returning reply")
            print(message)
            print(evaluation.feedback)
            reply = self.rerun(reply, message, history, evaluation.feedback, self.person.system_prompt)
        return reply

    def rerun(self, reply, message, history, feedback, system_prompt):
        updated_system_prompt = system_prompt + f"\n\n## Previous answer rejected\nYou just tried to reply, but the \
        quality control rejected your reply\n ## Your attempted answer:\n{reply}\n\n ## Reason \
        for rejection:\n{feedback}\n\n"
        messages = [
            {"role": "system", "content": updated_system_prompt}
                   ] + history + [
            {"role": "user", "content": message}
        ]
        response = self.llm.chat.completions.create(model=self.llm_model, messages=messages)
        return response.choices[0].message.content
