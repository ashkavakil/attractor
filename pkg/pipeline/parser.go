package pipeline

import (
	"fmt"
	"strconv"
	"strings"
	"time"
)

// Parser parses a DOT file into a pipeline Graph.
type Parser struct {
	tokens      []Token
	pos         int
	nodeDefaults map[string]string
	edgeDefaults map[string]string
}

// Parse parses DOT source into a pipeline.Graph.
func Parse(source string) (*Graph, error) {
	lexer := NewLexer(source)
	tokens, err := lexer.Tokenize()
	if err != nil {
		return nil, fmt.Errorf("lexer error: %w", err)
	}

	p := &Parser{
		tokens:       tokens,
		nodeDefaults: make(map[string]string),
		edgeDefaults: make(map[string]string),
	}
	return p.parseGraph()
}

func (p *Parser) peek() Token {
	if p.pos >= len(p.tokens) {
		return Token{Type: TokenEOF}
	}
	return p.tokens[p.pos]
}

func (p *Parser) advance() Token {
	tok := p.peek()
	if tok.Type != TokenEOF {
		p.pos++
	}
	return tok
}

func (p *Parser) expect(t TokenType) (Token, error) {
	tok := p.advance()
	if tok.Type != t {
		return tok, fmt.Errorf("expected %s but got %s (%q) at line %d, column %d",
			t, tok.Type, tok.Value, tok.Line, tok.Column)
	}
	return tok, nil
}

func (p *Parser) parseGraph() (*Graph, error) {
	if _, err := p.expect(TokenDigraph); err != nil {
		return nil, fmt.Errorf("expected 'digraph': %w", err)
	}

	// Graph name (optional)
	name := ""
	if p.peek().Type == TokenIdentifier || p.peek().Type == TokenString {
		name = p.advance().Value
	}

	if _, err := p.expect(TokenLBrace); err != nil {
		return nil, err
	}

	graph := &Graph{
		Name:  name,
		Nodes: make(map[string]*Node),
		Attrs: make(map[string]string),
	}

	if err := p.parseStatements(graph, nil); err != nil {
		return nil, err
	}

	if _, err := p.expect(TokenRBrace); err != nil {
		return nil, err
	}

	// Apply defaults and resolve graph attributes.
	p.resolveGraphAttrs(graph)

	return graph, nil
}

func (p *Parser) parseStatements(graph *Graph, subgraphDefaults map[string]string) error {
	for p.peek().Type != TokenRBrace && p.peek().Type != TokenEOF {
		if err := p.parseStatement(graph, subgraphDefaults); err != nil {
			return err
		}
	}
	return nil
}

func (p *Parser) parseStatement(graph *Graph, subgraphDefaults map[string]string) error {
	tok := p.peek()

	switch tok.Type {
	case TokenSemicolon:
		p.advance()
		return nil

	case TokenGraph:
		return p.parseGraphAttrStmt(graph)

	case TokenNode:
		return p.parseNodeDefaults()

	case TokenEdge:
		return p.parseEdgeDefaults()

	case TokenSubgraph:
		return p.parseSubgraph(graph)

	case TokenIdentifier, TokenString:
		return p.parseNodeOrEdge(graph, subgraphDefaults)

	default:
		return fmt.Errorf("unexpected token %s (%q) at line %d, column %d",
			tok.Type, tok.Value, tok.Line, tok.Column)
	}
}

func (p *Parser) parseGraphAttrStmt(graph *Graph) error {
	p.advance() // consume 'graph'

	if p.peek().Type == TokenLBracket {
		attrs, err := p.parseAttrBlock()
		if err != nil {
			return err
		}
		for k, v := range attrs {
			graph.Attrs[k] = v
		}
	} else if p.peek().Type == TokenEquals {
		// graph level: key = value (but this would be caught by identifier case)
	}

	p.skipSemicolon()
	return nil
}

func (p *Parser) parseNodeDefaults() error {
	p.advance() // consume 'node'
	attrs, err := p.parseAttrBlock()
	if err != nil {
		return err
	}
	for k, v := range attrs {
		p.nodeDefaults[k] = v
	}
	p.skipSemicolon()
	return nil
}

func (p *Parser) parseEdgeDefaults() error {
	p.advance() // consume 'edge'
	attrs, err := p.parseAttrBlock()
	if err != nil {
		return err
	}
	for k, v := range attrs {
		p.edgeDefaults[k] = v
	}
	p.skipSemicolon()
	return nil
}

func (p *Parser) parseSubgraph(graph *Graph) error {
	p.advance() // consume 'subgraph'

	// Optional subgraph name
	subgraphLabel := ""
	if p.peek().Type == TokenIdentifier || p.peek().Type == TokenString {
		subgraphLabel = p.advance().Value
	}

	if _, err := p.expect(TokenLBrace); err != nil {
		return err
	}

	// Save current defaults.
	savedNodeDefaults := make(map[string]string)
	for k, v := range p.nodeDefaults {
		savedNodeDefaults[k] = v
	}

	// Parse subgraph-local statements.
	sgDefaults := make(map[string]string)

	for p.peek().Type != TokenRBrace && p.peek().Type != TokenEOF {
		tok := p.peek()
		switch tok.Type {
		case TokenNode:
			if err := p.parseNodeDefaults(); err != nil {
				return err
			}
		case TokenGraph:
			// subgraph graph attrs
			p.advance()
			if p.peek().Type == TokenLBracket {
				attrs, err := p.parseAttrBlock()
				if err != nil {
					return err
				}
				for k, v := range attrs {
					sgDefaults[k] = v
				}
			}
			p.skipSemicolon()
		default:
			if err := p.parseStatement(graph, sgDefaults); err != nil {
				return err
			}
		}
	}

	if _, err := p.expect(TokenRBrace); err != nil {
		return err
	}

	// Derive class from subgraph label for nodes.
	if subgraphLabel != "" {
		derivedClass := deriveClassName(subgraphLabel)
		// Apply derived class to nodes that were defined in this subgraph and don't have explicit class.
		_ = derivedClass // Applied during node creation via subgraphDefaults
	}

	// Restore defaults.
	p.nodeDefaults = savedNodeDefaults

	p.skipSemicolon()
	return nil
}

// deriveClassName derives a CSS-like class name from a label.
func deriveClassName(label string) string {
	// Remove "cluster_" prefix if present.
	label = strings.TrimPrefix(label, "cluster_")
	label = strings.ToLower(label)
	label = strings.ReplaceAll(label, " ", "-")

	var result strings.Builder
	for _, r := range label {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '-' {
			result.WriteRune(r)
		}
	}
	return result.String()
}

func (p *Parser) parseNodeOrEdge(graph *Graph, subgraphDefaults map[string]string) error {
	// Read the first identifier.
	idTok := p.advance()
	id := idTok.Value

	// Check if this is a top-level key=value declaration.
	if p.peek().Type == TokenEquals {
		p.advance() // consume '='
		valueTok := p.advance()
		graph.Attrs[id] = valueTok.Value
		p.skipSemicolon()
		return nil
	}

	// Check if this is an edge statement (A -> B -> C).
	if p.peek().Type == TokenArrow {
		return p.parseEdgeChain(graph, id, subgraphDefaults)
	}

	// Otherwise it's a node statement.
	p.ensureNode(graph, id, subgraphDefaults)

	if p.peek().Type == TokenLBracket {
		attrs, err := p.parseAttrBlock()
		if err != nil {
			return err
		}
		p.applyNodeAttrs(graph.Nodes[id], attrs)
	}

	p.skipSemicolon()
	return nil
}

func (p *Parser) parseEdgeChain(graph *Graph, firstID string, subgraphDefaults map[string]string) error {
	chain := []string{firstID}

	for p.peek().Type == TokenArrow {
		p.advance() // consume '->'
		idTok := p.advance()
		chain = append(chain, idTok.Value)
	}

	// Parse optional edge attributes.
	var attrs map[string]string
	if p.peek().Type == TokenLBracket {
		var err error
		attrs, err = p.parseAttrBlock()
		if err != nil {
			return err
		}
	}

	// Create edges for each consecutive pair.
	for i := 0; i < len(chain)-1; i++ {
		from := chain[i]
		to := chain[i+1]

		p.ensureNode(graph, from, subgraphDefaults)
		p.ensureNode(graph, to, subgraphDefaults)

		edge := &Edge{
			From: from,
			To:   to,
		}

		// Apply edge defaults.
		for k, v := range p.edgeDefaults {
			p.applyEdgeAttr(edge, k, v)
		}

		// Apply explicit attrs.
		for k, v := range attrs {
			p.applyEdgeAttr(edge, k, v)
		}

		graph.Edges = append(graph.Edges, edge)
	}

	p.skipSemicolon()
	return nil
}

func (p *Parser) parseAttrBlock() (map[string]string, error) {
	if _, err := p.expect(TokenLBracket); err != nil {
		return nil, err
	}

	attrs := make(map[string]string)

	for p.peek().Type != TokenRBracket && p.peek().Type != TokenEOF {
		// Key (may be qualified: key.subkey)
		keyTok := p.advance()
		key := keyTok.Value

		// Handle qualified IDs (key.subkey.subsubkey)
		for p.peek().Type == TokenDot {
			p.advance() // consume '.'
			subkey := p.advance()
			key += "." + subkey.Value
		}

		if _, err := p.expect(TokenEquals); err != nil {
			return nil, err
		}

		// Value
		valTok := p.advance()
		attrs[key] = valTok.Value

		// Optional comma separator.
		if p.peek().Type == TokenComma {
			p.advance()
		}
	}

	if _, err := p.expect(TokenRBracket); err != nil {
		return nil, err
	}

	return attrs, nil
}

func (p *Parser) ensureNode(graph *Graph, id string, subgraphDefaults map[string]string) {
	if _, exists := graph.Nodes[id]; exists {
		return
	}

	node := &Node{
		ID:    id,
		Label: id,
		Shape: "box",
		Attrs: make(map[string]string),
	}

	// Apply node defaults.
	for k, v := range p.nodeDefaults {
		p.applyNodeAttr(node, k, v)
	}

	// Apply subgraph defaults.
	if subgraphDefaults != nil {
		for k, v := range subgraphDefaults {
			if k == "label" {
				// Subgraph label doesn't override node label
				continue
			}
			node.Attrs[k] = v
		}
	}

	graph.Nodes[id] = node
}

func (p *Parser) applyNodeAttrs(node *Node, attrs map[string]string) {
	for k, v := range attrs {
		p.applyNodeAttr(node, k, v)
	}
}

func (p *Parser) applyNodeAttr(node *Node, key, value string) {
	switch key {
	case "label":
		node.Label = value
	case "shape":
		node.Shape = value
	case "type":
		node.Type = value
	case "prompt":
		node.Prompt = value
	case "max_retries":
		n, _ := strconv.Atoi(value)
		node.MaxRetries = n
	case "goal_gate":
		node.GoalGate = value == "true"
	case "retry_target":
		node.RetryTarget = value
	case "fallback_retry_target":
		node.FallbackRetryTarget = value
	case "fidelity":
		node.Fidelity = value
	case "thread_id":
		node.ThreadID = value
	case "class":
		node.Class = value
	case "timeout":
		node.Timeout = parseDuration(value)
	case "llm_model":
		node.LLMModel = value
	case "llm_provider":
		node.LLMProvider = value
	case "reasoning_effort":
		node.ReasoningEffort = value
	case "auto_status":
		node.AutoStatus = value == "true"
	case "allow_partial":
		node.AllowPartial = value == "true"
	default:
		node.Attrs[key] = value
	}
}

func (p *Parser) applyEdgeAttr(edge *Edge, key, value string) {
	switch key {
	case "label":
		edge.Label = value
	case "condition":
		edge.Condition = value
	case "weight":
		n, _ := strconv.Atoi(value)
		edge.Weight = n
	case "fidelity":
		edge.Fidelity = value
	case "thread_id":
		edge.ThreadID = value
	case "loop_restart":
		edge.LoopRestart = value == "true"
	}
}

func (p *Parser) resolveGraphAttrs(graph *Graph) {
	if v, ok := graph.Attrs["goal"]; ok {
		graph.Goal = v
	}
	if v, ok := graph.Attrs["label"]; ok {
		graph.Label = v
	}
	if v, ok := graph.Attrs["model_stylesheet"]; ok {
		graph.ModelStylesheet = v
	}
	if v, ok := graph.Attrs["default_max_retry"]; ok {
		n, _ := strconv.Atoi(v)
		graph.DefaultMaxRetry = n
	}
	if v, ok := graph.Attrs["default_fidelity"]; ok {
		graph.DefaultFidelity = v
	}
	if v, ok := graph.Attrs["retry_target"]; ok {
		graph.RetryTarget = v
	}
	if v, ok := graph.Attrs["fallback_retry_target"]; ok {
		graph.FallbackRetryTarget = v
	}
}

func (p *Parser) skipSemicolon() {
	if p.peek().Type == TokenSemicolon {
		p.advance()
	}
}

// parseDuration parses duration strings like "900s", "15m", "2h", "250ms", "1d".
func parseDuration(s string) time.Duration {
	// Try standard Go duration first.
	d, err := time.ParseDuration(s)
	if err == nil {
		return d
	}

	// Handle 'd' suffix for days.
	if strings.HasSuffix(s, "d") {
		numStr := strings.TrimSuffix(s, "d")
		n, err := strconv.Atoi(numStr)
		if err == nil {
			return time.Duration(n) * 24 * time.Hour
		}
	}

	return 0
}
