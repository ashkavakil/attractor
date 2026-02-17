package agent

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/ashka-vakil/attractor/pkg/llm"
)

// Session is the central orchestrator for the coding agent loop.
type Session struct {
	ID              string
	ProviderProfile *ProviderProfile
	ExecutionEnv    ExecutionEnvironment
	History         []Turn
	EventEmitter    *EventEmitter
	Config          SessionConfig
	State           SessionState
	LLMClient       *llm.Client
	SteeringQueue   []string
	FollowupQueue   []string
	Subagents       map[string]*SubAgent

	mu              sync.Mutex
	turnCount       int
	loopDetector    *loopDetector
}

// NewSession creates a new agent session.
func NewSession(client *llm.Client, profile *ProviderProfile, env ExecutionEnvironment, config SessionConfig) *Session {
	if env == nil {
		env = NewLocalEnvironment()
	}
	s := &Session{
		ID:              fmt.Sprintf("session-%d", time.Now().UnixNano()),
		ProviderProfile: profile,
		ExecutionEnv:    env,
		EventEmitter:    NewEventEmitter(),
		Config:          config,
		State:           StateIdle,
		LLMClient:       client,
		Subagents:       make(map[string]*SubAgent),
		loopDetector:    newLoopDetector(config.LoopDetectionWindow),
	}
	return s
}

// Submit sends user input to the agent and processes it through the agentic loop.
func (s *Session) Submit(ctx context.Context, input string) error {
	s.mu.Lock()
	if s.State != StateIdle && s.State != StateAwaitingInput {
		s.mu.Unlock()
		return fmt.Errorf("session is not idle (state: %s)", s.State)
	}
	s.State = StateProcessing
	s.mu.Unlock()

	defer func() {
		s.mu.Lock()
		if s.State == StateProcessing {
			s.State = StateIdle
		}
		s.mu.Unlock()
	}()

	s.EventEmitter.Emit(Event{
		Type:      EventSessionStarted,
		Timestamp: time.Now(),
		Data:      map[string]interface{}{"input": input},
	})

	// Add user turn
	s.History = append(s.History, &UserTurn{
		Content:   input,
		Timestamp: time.Now(),
	})

	return s.runLoop(ctx)
}

// Steer injects a message between tool rounds.
func (s *Session) Steer(message string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.SteeringQueue = append(s.SteeringQueue, message)
}

// FollowUp queues a message for after the current input completes.
func (s *Session) FollowUp(message string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.FollowupQueue = append(s.FollowupQueue, message)
}

// Close terminates the session.
func (s *Session) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.State = StateClosed
	s.EventEmitter.Emit(Event{
		Type:      EventSessionClosed,
		Timestamp: time.Now(),
	})
}

// runLoop is the core agentic loop.
func (s *Session) runLoop(ctx context.Context) error {
	toolRound := 0
	maxToolRounds := s.Config.MaxToolRoundsPerInput
	if maxToolRounds == 0 {
		maxToolRounds = 200 // safety limit
	}

	for toolRound < maxToolRounds {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Check for steering messages
		s.mu.Lock()
		if len(s.SteeringQueue) > 0 {
			msg := s.SteeringQueue[0]
			s.SteeringQueue = s.SteeringQueue[1:]
			s.mu.Unlock()

			s.History = append(s.History, &SteeringTurn{
				Content:   msg,
				Timestamp: time.Now(),
			})
			s.EventEmitter.Emit(Event{
				Type:      EventSteeringApplied,
				Timestamp: time.Now(),
				Data:      map[string]interface{}{"message": msg},
			})
		} else {
			s.mu.Unlock()
		}

		// Build LLM request from history
		req := s.buildRequest()

		s.EventEmitter.Emit(Event{
			Type:      EventTurnStarted,
			Timestamp: time.Now(),
			Data:      map[string]interface{}{"tool_round": toolRound},
		})

		// Call LLM
		resp, err := s.LLMClient.Complete(ctx, req)
		if err != nil {
			s.EventEmitter.Emit(Event{
				Type:      EventError,
				Timestamp: time.Now(),
				Data:      map[string]interface{}{"error": err.Error()},
			})
			return fmt.Errorf("LLM call failed: %w", err)
		}

		// Record assistant turn
		assistantTurn := &AssistantTurn{
			Content:    resp.Content,
			ToolCalls:  resp.ToolCalls,
			Reasoning:  resp.Reasoning,
			Usage:      resp.Usage,
			ResponseID: resp.ID,
			Timestamp:  time.Now(),
		}
		s.History = append(s.History, assistantTurn)
		s.turnCount++

		// Check turn limit
		if s.Config.MaxTurns > 0 && s.turnCount >= s.Config.MaxTurns {
			break
		}

		// If no tool calls, the loop is done
		if len(resp.ToolCalls) == 0 {
			s.EventEmitter.Emit(Event{
				Type:      EventTurnCompleted,
				Timestamp: time.Now(),
				Data: map[string]interface{}{
					"content":    resp.Content,
					"tool_round": toolRound,
				},
			})
			break
		}

		// Execute tool calls
		results, err := s.executeToolCalls(ctx, resp.ToolCalls)
		if err != nil {
			return fmt.Errorf("tool execution failed: %w", err)
		}

		// Record tool results
		s.History = append(s.History, &ToolResultsTurn{
			Results:   results,
			Timestamp: time.Now(),
		})

		// Loop detection
		if s.Config.EnableLoopDetection {
			for _, tc := range resp.ToolCalls {
				if s.loopDetector.recordAndCheck(tc.Name, string(tc.Arguments)) {
					s.EventEmitter.Emit(Event{
						Type:      EventLoopDetected,
						Timestamp: time.Now(),
						Data: map[string]interface{}{
							"tool":  tc.Name,
							"count": s.Config.LoopDetectionWindow,
						},
					})
					// Inject steering to break the loop
					s.Steer("You appear to be in a loop calling the same tool repeatedly. Please try a different approach.")
				}
			}
		}

		toolRound++
	}

	// Process follow-up queue
	s.mu.Lock()
	if len(s.FollowupQueue) > 0 {
		msg := s.FollowupQueue[0]
		s.FollowupQueue = s.FollowupQueue[1:]
		s.mu.Unlock()
		return s.Submit(ctx, msg)
	}
	s.mu.Unlock()

	return nil
}

func (s *Session) buildRequest() *llm.Request {
	req := &llm.Request{
		Model: s.ProviderProfile.Model,
	}

	if s.ProviderProfile.SystemPrompt != "" {
		req.SystemPrompt = s.ProviderProfile.SystemPrompt
	}

	if s.Config.ReasoningEffort != "" {
		req.ReasoningEffort = s.Config.ReasoningEffort
	}

	// Build tools
	for _, t := range s.ProviderProfile.Tools {
		req.Tools = append(req.Tools, t)
	}

	// Build messages from history
	for _, turn := range s.History {
		switch t := turn.(type) {
		case *UserTurn:
			req.Messages = append(req.Messages, llm.Message{
				Role:    llm.RoleUser,
				Content: t.Content,
			})
		case *SteeringTurn:
			// Steering turns are sent as user messages to the LLM
			req.Messages = append(req.Messages, llm.Message{
				Role:    llm.RoleUser,
				Content: t.Content,
			})
		case *AssistantTurn:
			msg := llm.Message{
				Role:      llm.RoleAssistant,
				Content:   t.Content,
				ToolCalls: t.ToolCalls,
			}
			req.Messages = append(req.Messages, msg)
		case *ToolResultsTurn:
			for _, r := range t.Results {
				req.Messages = append(req.Messages, llm.Message{
					Role:       llm.RoleTool,
					Content:    r.Content,
					ToolCallID: r.ToolCallID,
				})
			}
		}
	}

	return req
}

func (s *Session) executeToolCalls(ctx context.Context, toolCalls []llm.ToolCall) ([]llm.ToolResult, error) {
	results := make([]llm.ToolResult, len(toolCalls))
	for i, tc := range toolCalls {
		s.EventEmitter.Emit(Event{
			Type:      EventToolCallStarted,
			Timestamp: time.Now(),
			Data: map[string]interface{}{
				"tool_name": tc.Name,
				"tool_id":   tc.ID,
			},
		})

		result, err := s.ExecutionEnv.Execute(ctx, tc.Name, tc.Arguments)
		if err != nil {
			results[i] = llm.ToolResult{
				ToolCallID: tc.ID,
				Content:    fmt.Sprintf("Error: %s", err),
				IsError:    true,
			}
		} else {
			// Apply full two-stage truncation pipeline
			content := s.applyTruncation(tc.Name, result)

			results[i] = llm.ToolResult{
				ToolCallID: tc.ID,
				Content:    content,
			}
		}

		// TOOL_CALL_END event carries full untruncated output
		s.EventEmitter.Emit(Event{
			Type:      EventToolCallCompleted,
			Timestamp: time.Now(),
			Data: map[string]interface{}{
				"tool_name": tc.Name,
				"tool_id":   tc.ID,
				"is_error":  results[i].IsError,
				"output":    result, // full untruncated output
			},
		})
	}
	return results, nil
}

// defaultToolOutputLimits provides default character limits per tool name.
var defaultToolOutputLimits = map[string]int{
	"read_file": 50000,
	"bash":      30000,
	"grep":      20000,
	"glob":      10000,
}

// getToolOutputLimit returns the effective output limit for a tool.
func (s *Session) getToolOutputLimit(toolName string) int {
	if limit, ok := s.Config.ToolOutputLimits[toolName]; ok {
		return limit
	}
	if limit, ok := defaultToolOutputLimits[toolName]; ok {
		return limit
	}
	return 50000 // default fallback
}

// defaultLineLimits provides default line limits per tool (applied after char truncation).
var defaultLineLimits = map[string]int{
	"bash": 256,
	"grep": 200,
	"glob": 500,
}

// truncateToolOutput applies character-based head/tail truncation with a visible warning marker.
// This runs FIRST. Line-based truncation runs second via truncateLines.
func truncateToolOutput(content string, limit int) string {
	if len(content) <= limit {
		return content
	}
	headSize := limit * 3 / 4
	tailSize := limit - headSize - 150 // room for truncation message

	if tailSize < 0 {
		tailSize = 0
	}

	head := content[:headSize]
	truncated := len(content) - headSize - tailSize
	marker := fmt.Sprintf("\n\n[WARNING: Tool output was truncated. %d characters removed from the middle. The full output is available in the event stream. If you need to see specific parts, re-run with more targeted parameters.]\n\n", truncated)

	if tailSize > 0 {
		tail := content[len(content)-tailSize:]
		return head + marker + tail
	}
	return head + marker
}

// truncateLines applies line-based head/tail truncation as a second pass.
func truncateLines(content string, maxLines int) string {
	lines := strings.Split(content, "\n")
	if len(lines) <= maxLines {
		return content
	}

	headCount := maxLines / 2
	tailCount := maxLines - headCount
	omitted := len(lines) - headCount - tailCount

	head := strings.Join(lines[:headCount], "\n")
	tail := strings.Join(lines[len(lines)-tailCount:], "\n")
	return head + fmt.Sprintf("\n[... %d lines omitted ...]\n", omitted) + tail
}

// applyTruncation applies the full two-stage truncation pipeline to tool output.
func (s *Session) applyTruncation(toolName, output string) string {
	// Stage 1: Character-based truncation (always runs first)
	charLimit := s.getToolOutputLimit(toolName)
	result := truncateToolOutput(output, charLimit)

	// Stage 2: Line-based truncation (secondary)
	if lineLimit, ok := defaultLineLimits[toolName]; ok {
		result = truncateLines(result, lineLimit)
	}

	return result
}

// --- Loop Detector ---

type loopDetector struct {
	window  int
	history []string
}

func newLoopDetector(window int) *loopDetector {
	return &loopDetector{window: window}
}

func (ld *loopDetector) recordAndCheck(name, args string) bool {
	key := name + ":" + args
	ld.history = append(ld.history, key)

	if len(ld.history) < ld.window {
		return false
	}

	// Check for repeating patterns of length 1, 2, or 3
	recent := ld.history[len(ld.history)-ld.window:]
	for _, patternLen := range []int{1, 2, 3} {
		if ld.window%patternLen != 0 {
			continue
		}
		pattern := recent[:patternLen]
		allMatch := true
		for i := patternLen; i < ld.window; i += patternLen {
			end := i + patternLen
			if end > ld.window {
				break
			}
			chunk := recent[i:end]
			for j := 0; j < patternLen; j++ {
				if chunk[j] != pattern[j] {
					allMatch = false
					break
				}
			}
			if !allMatch {
				break
			}
		}
		if allMatch {
			return true
		}
	}
	return false
}
