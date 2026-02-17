package pipeline

import (
	"fmt"
	"strings"
	"unicode"
)

// TokenType identifies the type of a lexer token.
type TokenType int

const (
	TokenEOF TokenType = iota
	TokenDigraph
	TokenSubgraph
	TokenGraph
	TokenNode
	TokenEdge
	TokenIdentifier
	TokenString
	TokenInteger
	TokenFloat
	TokenBoolean
	TokenLBrace
	TokenRBrace
	TokenLBracket
	TokenRBracket
	TokenEquals
	TokenComma
	TokenSemicolon
	TokenArrow
	TokenDot
)

var tokenNames = map[TokenType]string{
	TokenEOF:        "EOF",
	TokenDigraph:    "digraph",
	TokenSubgraph:   "subgraph",
	TokenGraph:      "graph",
	TokenNode:       "node",
	TokenEdge:       "edge",
	TokenIdentifier: "identifier",
	TokenString:     "string",
	TokenInteger:    "integer",
	TokenFloat:      "float",
	TokenBoolean:    "boolean",
	TokenLBrace:     "{",
	TokenRBrace:     "}",
	TokenLBracket:   "[",
	TokenRBracket:   "]",
	TokenEquals:     "=",
	TokenComma:      ",",
	TokenSemicolon:  ";",
	TokenArrow:      "->",
	TokenDot:        ".",
}

func (t TokenType) String() string {
	if s, ok := tokenNames[t]; ok {
		return s
	}
	return fmt.Sprintf("token(%d)", int(t))
}

// Token is a lexer token.
type Token struct {
	Type   TokenType
	Value  string
	Line   int
	Column int
}

// Lexer tokenizes DOT source.
type Lexer struct {
	input  []rune
	pos    int
	line   int
	col    int
	tokens []Token
}

// NewLexer creates a new lexer for the given input.
func NewLexer(input string) *Lexer {
	// Strip comments first.
	cleaned := stripComments(input)
	return &Lexer{
		input: []rune(cleaned),
		line:  1,
		col:   1,
	}
}

// stripComments removes // and /* */ comments.
func stripComments(s string) string {
	var result strings.Builder
	runes := []rune(s)
	i := 0
	inString := false
	for i < len(runes) {
		if inString {
			if runes[i] == '\\' && i+1 < len(runes) {
				result.WriteRune(runes[i])
				result.WriteRune(runes[i+1])
				i += 2
				continue
			}
			if runes[i] == '"' {
				inString = false
			}
			result.WriteRune(runes[i])
			i++
			continue
		}

		if runes[i] == '"' {
			inString = true
			result.WriteRune(runes[i])
			i++
			continue
		}

		// Line comment
		if runes[i] == '/' && i+1 < len(runes) && runes[i+1] == '/' {
			for i < len(runes) && runes[i] != '\n' {
				i++
			}
			continue
		}

		// Block comment
		if runes[i] == '/' && i+1 < len(runes) && runes[i+1] == '*' {
			i += 2
			for i+1 < len(runes) {
				if runes[i] == '*' && runes[i+1] == '/' {
					i += 2
					break
				}
				if runes[i] == '\n' {
					result.WriteRune('\n') // preserve line numbers
				}
				i++
			}
			continue
		}

		result.WriteRune(runes[i])
		i++
	}
	return result.String()
}

// Tokenize produces all tokens from the input.
func (l *Lexer) Tokenize() ([]Token, error) {
	for {
		tok, err := l.next()
		if err != nil {
			return nil, err
		}
		l.tokens = append(l.tokens, tok)
		if tok.Type == TokenEOF {
			break
		}
	}
	return l.tokens, nil
}

func (l *Lexer) peek() rune {
	if l.pos >= len(l.input) {
		return 0
	}
	return l.input[l.pos]
}

func (l *Lexer) advance() rune {
	if l.pos >= len(l.input) {
		return 0
	}
	r := l.input[l.pos]
	l.pos++
	if r == '\n' {
		l.line++
		l.col = 1
	} else {
		l.col++
	}
	return r
}

func (l *Lexer) skipWhitespace() {
	for l.pos < len(l.input) && unicode.IsSpace(l.input[l.pos]) {
		l.advance()
	}
}

func (l *Lexer) next() (Token, error) {
	l.skipWhitespace()

	if l.pos >= len(l.input) {
		return Token{Type: TokenEOF, Line: l.line, Column: l.col}, nil
	}

	startLine := l.line
	startCol := l.col
	ch := l.peek()

	// Single character tokens
	switch ch {
	case '{':
		l.advance()
		return Token{Type: TokenLBrace, Value: "{", Line: startLine, Column: startCol}, nil
	case '}':
		l.advance()
		return Token{Type: TokenRBrace, Value: "}", Line: startLine, Column: startCol}, nil
	case '[':
		l.advance()
		return Token{Type: TokenLBracket, Value: "[", Line: startLine, Column: startCol}, nil
	case ']':
		l.advance()
		return Token{Type: TokenRBracket, Value: "]", Line: startLine, Column: startCol}, nil
	case '=':
		l.advance()
		return Token{Type: TokenEquals, Value: "=", Line: startLine, Column: startCol}, nil
	case ',':
		l.advance()
		return Token{Type: TokenComma, Value: ",", Line: startLine, Column: startCol}, nil
	case ';':
		l.advance()
		return Token{Type: TokenSemicolon, Value: ";", Line: startLine, Column: startCol}, nil
	case '.':
		l.advance()
		return Token{Type: TokenDot, Value: ".", Line: startLine, Column: startCol}, nil
	}

	// Arrow ->
	if ch == '-' && l.pos+1 < len(l.input) && l.input[l.pos+1] == '>' {
		l.advance()
		l.advance()
		return Token{Type: TokenArrow, Value: "->", Line: startLine, Column: startCol}, nil
	}

	// String
	if ch == '"' {
		return l.readString(startLine, startCol)
	}

	// Number (including negative)
	if ch == '-' || unicode.IsDigit(ch) {
		return l.readNumber(startLine, startCol)
	}

	// Identifier or keyword
	if unicode.IsLetter(ch) || ch == '_' {
		return l.readIdentifier(startLine, startCol)
	}

	return Token{}, fmt.Errorf("unexpected character %q at line %d, column %d", ch, startLine, startCol)
}

func (l *Lexer) readString(line, col int) (Token, error) {
	l.advance() // skip opening quote
	var s strings.Builder
	for l.pos < len(l.input) {
		ch := l.advance()
		if ch == '\\' {
			next := l.advance()
			switch next {
			case '"':
				s.WriteByte('"')
			case 'n':
				s.WriteByte('\n')
			case 't':
				s.WriteByte('\t')
			case '\\':
				s.WriteByte('\\')
			default:
				s.WriteByte('\\')
				s.WriteRune(next)
			}
			continue
		}
		if ch == '"' {
			return Token{Type: TokenString, Value: s.String(), Line: line, Column: col}, nil
		}
		s.WriteRune(ch)
	}
	return Token{}, fmt.Errorf("unterminated string at line %d, column %d", line, col)
}

func (l *Lexer) readNumber(line, col int) (Token, error) {
	var s strings.Builder
	isFloat := false

	if l.peek() == '-' {
		s.WriteRune(l.advance())
	}

	// If the next char after '-' is not a digit, this is not a number
	if l.pos >= len(l.input) || !unicode.IsDigit(l.peek()) {
		// It was just a '-' which we shouldn't have consumed as a number.
		// But since we handle '->' separately, this shouldn't happen in valid input.
		return Token{}, fmt.Errorf("unexpected '-' at line %d, column %d", line, col)
	}

	for l.pos < len(l.input) && unicode.IsDigit(l.peek()) {
		s.WriteRune(l.advance())
	}

	if l.pos < len(l.input) && l.peek() == '.' {
		isFloat = true
		s.WriteRune(l.advance())
		for l.pos < len(l.input) && unicode.IsDigit(l.peek()) {
			s.WriteRune(l.advance())
		}
	}

	// Check for duration suffix
	if l.pos < len(l.input) && unicode.IsLetter(l.peek()) {
		for l.pos < len(l.input) && unicode.IsLetter(l.peek()) {
			s.WriteRune(l.advance())
		}
		return Token{Type: TokenString, Value: s.String(), Line: line, Column: col}, nil
	}

	if isFloat {
		return Token{Type: TokenFloat, Value: s.String(), Line: line, Column: col}, nil
	}
	return Token{Type: TokenInteger, Value: s.String(), Line: line, Column: col}, nil
}

func (l *Lexer) readIdentifier(line, col int) (Token, error) {
	var s strings.Builder
	for l.pos < len(l.input) && (unicode.IsLetter(l.peek()) || unicode.IsDigit(l.peek()) || l.peek() == '_') {
		s.WriteRune(l.advance())
	}

	value := s.String()

	switch value {
	case "digraph":
		return Token{Type: TokenDigraph, Value: value, Line: line, Column: col}, nil
	case "subgraph":
		return Token{Type: TokenSubgraph, Value: value, Line: line, Column: col}, nil
	case "graph":
		return Token{Type: TokenGraph, Value: value, Line: line, Column: col}, nil
	case "node":
		return Token{Type: TokenNode, Value: value, Line: line, Column: col}, nil
	case "edge":
		return Token{Type: TokenEdge, Value: value, Line: line, Column: col}, nil
	case "true", "false":
		return Token{Type: TokenBoolean, Value: value, Line: line, Column: col}, nil
	default:
		return Token{Type: TokenIdentifier, Value: value, Line: line, Column: col}, nil
	}
}
