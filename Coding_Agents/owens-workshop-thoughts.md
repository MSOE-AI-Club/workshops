# Coding Agents Workshop Thoughts

I understand this workshop as being focused on using Antigravity with free access and helping beginner-intermediate participants understand agentic workflows. We should keep the explanation of what agents are and what tools are available so people understand how to use them.

## Main workflow

Spec-driven development should be a big part of this. Developing requirements and a plan is super useful because it gives the agent direction and gives us something to check its work against.

1. Develop clear requirements and acceptance criteria.
2. Use Plan Mode to explore the project and propose a plan.
3. Review the plan, question its assumptions, and revise it.
4. Have the agent implement the approved plan.
5. Verify the implementation with tests and browser checks where appropriate.
6. Do an adversarial review against the requirements (we should talk about what this is.)
7. Reproduce any findings, fix confirmed problems, and verify again.

We should give agents ways to check their work and iterate. Passing tests and an approved plan can still miss a requirement. Participants should review the evidence themselves.

## Exercises

We should do both a shared demo with a seeded bug and an exercise where participants review an agent-built feature.

For the demo, normal behavior should work and the tests should pass, but an edge case should violate a requirement. Let participants judge the evidence before revealing that we included a flaw.

For the participant exercise, they should follow the same workflow. The reviewer should look for concrete failures and explain how to reproduce them. Finding no confirmed issue is also a valid result.

## Starter repo

Participants could clone a repo with a small working app, setup instructions, verification commands, and prebuilt skills for plan review, verification, and adversarial review.

They should read the skills, use them, and improve an instruction based on what they observe. We should keep the scaffolding simple and leave meaningful requirements and review decisions for participants to work through. It also would be good to have the participant make a new skill of their choice.

## Closing Thoughts

Those are my thoughts and what I think would give the participants the best results in the workshop. I think defining the problem they will be solving is important and that we try and use some of the features that Antigravity is great at. Maybe a full-stack site? Show off the frontend features and browser testing that Antigravity does well.

## Thoughts from the club

take home stuff
make skills repo
project
look at a simple project to do.
look at teams thing
