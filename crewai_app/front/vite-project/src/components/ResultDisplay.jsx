import styled from 'styled-components';
import { FaDownload } from 'react-icons/fa';

const ResultContainer = styled.div`
  margin-top: 20px;
  text-align: center;
  padding: 20px;
  background-color: #f9f9f9;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
`;

const ResultText = styled.pre`
  background-color: #f8f8f8;
  padding: 15px;
  border-radius: 8px;
  white-space: pre-wrap;
  text-align: left;
  max-height: 400px;
  overflow-y: auto;
  margin-bottom: 20px;
`;

const DownloadButton = styled.button`
  padding: 12px 24px;
  background-color: #28a745;
  color: #fff;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 1rem;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background-color 0.3s ease;
  &:hover {
    background-color: #218828;
  }
`;

const ResultDisplay = () => {
  return (
    <ResultContainer>
      <h2>작성한 콘텐츠를 다운로드해 보세요.</h2>
      <ResultText>작성된 블로그 글</ResultText>
      <DownloadButton>
        <FaDownload style={{ marginRight: '8px' }} />
        다운로드
      </DownloadButton>
    </ResultContainer>
  );
};

export default ResultDisplay;
